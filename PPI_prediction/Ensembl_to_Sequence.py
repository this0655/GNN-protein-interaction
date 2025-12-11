import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry

"""
Get uniprot id and sequence with ensembl id using uniprot api
input file: ensembl id list
output file: ensembl id, uniprot id, sequence file
"""

POLLING_INTERVAL = 3   # 결과가 아직 준비되지 않았을 때 재시도 간격
API_URL = "https://rest.uniprot.org"   # UniProt의 공식 REST API 엔드포인트(root URL) 를 상수로 지정한 것


retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
# 5번 재시도, 재시도 간격은 0.25초씩 증가, 특정 HTTP 상태 코드(error code)에 대해 재시도
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))
# requests.adapters.HTTPAdapter를 통해 방금 정의한 retries 정책을 HTTPS 프로토콜에 전역 적용


def check_response(response):
    try:
        response.raise_for_status()   # HTTP 오류(400, 500번대)가 발생하면 예외(request.HTTPError) 발생
    except requests.HTTPError:
        # JSON이 아닐 수도 있으므로 안전하게 처리
        try:
            print(response.json())
        except ValueError:
            print(response.text)   # JSON 파싱 실패 시 그냥 텍스트 그대로 출력
        raise   # 예외를 상위 호출 함수로 다시 전달


def submit_id_mapping(from_db, to_db, ids):

    # submit_id_mapping() 함수는 UniProt의 ID Mapping API를 실행(job 제출) 하는 부분
    # 즉, “Ensembl ID → UniProt ID”, “UniProt → RefSeq” 같은 변환 작업을 UniProt 서버에 요청해서 매핑 작업(job)을 시작하는 함수

    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        # from: 원본 DB "ensembl", to: 대상 DB "uniprotkb", ids: 변환할 ID 목록
    )
    check_response(request)
    return request.json()["jobId"]   # 생성된 매핑 Job ID 추출


def get_next_link(headers):

    # get_next_link()는 UniProt REST API가 여러 페이지로 나뉘어 결과를 반환할 때,
    # 그 다음 페이지(next page)의 URL을 자동으로 찾아주는 페이지네이션(pagination) 처리용 함수

    re_next_link = re.compile(r'<(.+)>; rel="next"')   #'Link' 헤더 안에서 rel="next" 패턴을 찾기 위한 패턴 객체 생성
    if "Link" in headers:   # 응답 결과가 여러 장이면 다음 페이지 링크가 헤더에 포함됨
        match = re_next_link.match(headers["Link"])   # "Link" 문자열에서 <URL>; rel="next" 패턴 찾기
        if match:
            return match.group(1)
    # headers["Link"] = '<https://rest.uniprot.org/idmapping/results/abcdef?page=2>; rel="next"'
    # match = re_next_link.match(headers["Link"])
    # match.group(1) → 'https://rest.uniprot.org/idmapping/results/abcdef?page=2'


def check_id_mapping_results_ready(job_id):

    # 1. 주어진 작업 ID(job_id) 에 대해 UniProt 서버의 상태를 계속 조회 (/idmapping/status/{job_id}).
    # 2. 상태가 "NEW" 또는 "RUNNING"이면 일정 시간(POLLING_INTERVAL) 대기 후 재시도.
    # 3. 상태가 "FINISHED"가 되면 결과가 준비되었다고 판단하고 True 반환.
    # 4. 만약 "FAILED", "ERROR", "CANCELLED" 등의 상태이면 예외(Exception) 발생.
    # 즉, 폴링(polling) 을 통해 “작업이 완료될 때까지 기다리는 함수”

    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")   # check_response() 함수를 호출해 응답 상태를 검사(200번대, 그 이외 => 에러)
        check_response(request)
        j = request.json()   # 서버로부터 받은 응답 본문을 JSON 형식으로 파싱하여 Python 딕셔너리로 변환.
        if "jobStatus" in j:   # 응답에 jobStatus가 있으면 작업이 아직 안끝나거나 완료 상태가 아님.
            if j["jobStatus"] in ("NEW", "RUNNING"):   #NEW: 작업이 대기열에 등록됨, RUNNING: 작업이 실행 중
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)   # 일정 시간(3초) 대기 후 루프의 처음으로 돌아가 다시 상태 조회
            else:
                raise Exception(j["jobStatus"])   #"FAILED", "ERROR", "CANCELLED" 등의 상태이면 예외 발생(작업의 비정상 종료)
        else:
            return bool(j["results"] or j["failedIds"])   # jobStatus가 없고 results나 failedIds가 있으면 결과가 준비된 것. True 반환


def decode_results(response, file_format, compressed):
    if compressed:   # 응답이 압축되어 있으면 바이너리 압축 데이터이므로 zlib로 해제
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))   # Python dict로 변환
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]   # 줄 단위로 분할 후 리스트로 반환
        elif file_format == "xlsx":
            return [decompressed]   # 엑셀은 바이너리 데이터 그대로 반환
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]   # XML 포맷은 텍스트이므로 UTF-8로 디코딩 후 문자열로 반환
        else:
            return decompressed.decode("utf-8")   # 단순히 텍스트(예: FASTA)로 간주하고 UTF-8 문자열로 반환
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text   # 형식이 명시되지 않은 일반 텍스트(주로 FASTA 또는 기타 텍스트 데이터)


def get_batch(batch_response, file_format, compressed):

    # UniProt의 /idmapping/results/{job_id} API는 결과를 페이지 단위(batch) 로 나눠서 보냄
    # 각 페이지가 끝날 때마다 HTTP 헤더에 "Link: <URL>; rel='next'"가 붙어 있으며, 다음 페이지의 URL을 가리킴
    # get_batch() 함수는 이 "next" 링크를 따라가며 모든 페이지를 자동으로 순회
    # 각 페이지를 불러올 때마다 decode_results()를 통해 내용(FASTA, TSV, JSON 등)을 해석하고, yield로 한 번에 하나씩 반환
    # 즉, 이 함수는 “모든 페이지를 자동으로 순회하며 결과를 순차적으로 디코딩해주는 generator”

    batch_url = get_next_link(batch_response.headers)   # 첫 번째 페이지의 헤더에서 "Link"필드를 찾아 다음 페이지 URL(rel="next") 추출
    while batch_url:   # 다음 페이지가 있으면 계속 반복
        batch_response = session.get(batch_url)   # 다음 batch 페이지를 실제로 요청(GET)
        batch_response.raise_for_status()   # 오류(HTTP 상태 코드가 400, 500번대)가 발생하면 request.HTTPError 예외 발생
        yield decode_results(batch_response, file_format, compressed)   # 응답 데이터를 decode_results() 함수를 통해 해석한 후, 그 결과를 한 번에 하나씩 내보냄
        batch_url = get_next_link(batch_response.headers)   # 다음 페이지가 있으면 계속 반복


def combine_batches(all_results, batch_results, file_format):

    # 이 함수는 각 batch 결과(batch_results)를 누적 결과(all_results)에 병합(combine)

    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:   # batch_results[key]가 비어있지 않거나, batch_results에 key가 있으면
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]   # 첫 번째 줄(헤더)을 제외하고 나머지 줄을 모두 합침
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):

    # UniProt의 ID 매핑은 비동기 작업이므로 job_id만으로는 직접 결과를 받을 수 없음
    # UniProt REST API는 idmapping/details/{job_id} 엔드포인트를 제공하여 이 작업(job)의 메타정보(상태, 크기, 결과 위치)를 알려줌
    # 여기서 반환되는 JSON 안의 "redirectURL"은 실제 결과가 있는 URL
    # 이 URL로 이동하면 실제 매핑된 결과(예: JSON, TSV, FASTA) 를 다운로드할 수 있음
    # 즉, “작업 ID → 실제 결과 URL”을 얻는 중간 단계

    url = f"{API_URL}/idmapping/details/{job_id}"   # UniProt REST API의 idmapping/details/{job_id} 엔드포인트 URL을 생성
    request = session.get(url)   # 위에서 만든 URL에 `GET` 요청을 보냄, session에는 재시도 정책이 적용되어 있음
    check_response(request)
    return request.json()["redirectURL"]   # 'request.json()'으로 응답 본문을 JSON 파싱.


def get_xml_namespace(element):

    # Uniprot XML 예시:
    # <{http://uniprot.org/uniprot}entry>
    # → 여기서 {http://uniprot.org/uniprot} 부분이 바로 XML namespace
    # 이 함수는 주어진 XML element의 tag 문자열에서 namespace 부분만 추출하여 반환
    # 즉, XML 요소 태그에서 {...} 안의 URL 부분(네임스페이스)을 추출하는 함수

    m = re.match(r"\{(.*)\}", element.tag)   # 정규표현식으로 element.tag에서 {}로 감싸진 부분을 찾음
    return m.groups()[0] if m else ""   # m이 None이면 "" 반환, 있으면 그룹 반환


def merge_xml_results(xml_results):

    # 입력:
    # 여러 개의 ElementTree.XML 문서(batch 결과들). (xml_results는 문자열 리스트, 각 요소가 <uniprot>...</uniprot> 형태의 XML 텍스트)
    # 출력:
    # 모든 batch 결과의 <entry>들을 합쳐서 하나의 XML 바이트열(bytes) 로 반환.

    # 즉, 여러 XML batch 문서들을 합쳐서, 단일 <uniprot> 루트 아래의 모든 <entry>를 통합하는 함수
    # 이건 주로 UniProt에서 XML 포맷(file_format="xml")으로 결과를 요청했을 때 사용

    merged_root = ElementTree.fromstring(xml_results[0])   # 첫 번째 XML 결과(문자열)를 ElementTree 객체로 파싱하여 루트 요소 추출(트리의 최상위 노드)
    for result in xml_results[1:]:
        root = ElementTree.fromstring(result)   #  # 두 번째 이후의 XML 결과(문자열)를 ElementTree 객체로 파싱하여 루트 요소 추출
        for child in root.findall("{http://uniprot.org/uniprot}entry"):   # 현재 XML 문서(`root`)에서 모든 `<entry>` 태그를 찾음
            merged_root.insert(-1, child)   # 찾은 `<entry>` 요소들을 `merged_root`의 마지막 바로 앞에 삽입(즉, 루트의 자식으로 추가)
    ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))   # 이 네임스페이스를 기본(default) 네임스페이스로 등록. 출력이 <entry>로 시작하도록 하기 위함
    return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)   # 병합된 전체 XML 트리를 문자열로 변환(bytes 반환)


def print_progress_batches(batch_index, size, total):

    # 이 함수는 UniProt 서버에서 결과를 여러 batch(page) 단위로 가져올 때, 지금까지 몇 개의 결과를 가져왔는지 콘솔에 출력

    n_fetched = min((batch_index + 1) * size, total)   # (batch_index + 1) * size: 지금까지 가져온 결과 개수(batch는 한번 가져오는 단위, size는 한 batch의 크기)
    print(f"Fetched: {n_fetched} / {total}")


def get_id_mapping_results_search(url):

    # 이 함수는 UniProt의 /idmapping/results/{job_id} 혹은 /idmapping/results/{job_id}?format=... 
    # URL을 받아서 모든 결과 데이터를 자동으로 가져오고, 병합하며, 완성된 결과를 반환

    parsed = urlparse(url)   # 표준 라이브러리 urllib.parse.urlparse()를 이용해 URL을 구조적으로 분해
    # ParseResult(
    #     scheme='https',
    #     netloc='rest.uniprot.org',
    #     path='/idmapping/results/da4b9237...',
    #     params='',
    #     query='format=json&size=500',
    #     fragment=''
    # )
    query = parse_qs(parsed.query)   # URL의 쿼리 문자열(`format=json&size=500`)을 파싱하여 딕셔너리로 변환
    file_format = query["format"][0] if "format" in query else "json"   # query에 format이 명시되지 않으면 기본값은 "json"
    if "size" in query:
        size = int(query["size"][0])   # query에 size가 명시되어 있으면 그 값을 정수로 변환
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
        # 전체 삼항 표현식 (A if C else B)
        # 평가 순서: 먼저 C("compressed" in query)를 평가. True면 A(위의 비교)를 평가·반환, False면 B(False)를 반환.
        # 즉, query에 "compressed"가 없으면 False, 있으면 그 값을 소문자로 바꿔서 "true"와 비교해서 맞으면 True, 맞지 않으면 False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))   # doseq=True는 리스트 값을 여러 개의 쿼리 파라미터로 변환할 때 사용
    url = parsed.geturl()   # 변경된 쿼리 딕셔너리를 다시 URL 문자열로 재조합
    request = session.get(url)   # 위에서 만든 URL에 `GET` 요청을 보냄
    check_response(request)   # 응답 상태 코드 확인
    results = decode_results(request, file_format, compressed)   # 응답 본문(request)을 지정된 포맷(file_format)과 압축 설정(compressed)에 맞게 디코딩
    total = int(request.headers["x-total-results"])   # HTTP 응답 헤더에서 전체 결과 개수(`x-total-results`)를 가져와 정수로 변환
    print_progress_batches(0, size, total)   # 진행 상황 출력
    for i, batch in enumerate(get_batch(request, file_format, compressed), 1):   # get_batch() 제너레이터를 통해 다음 batch 결과들을 순차적으로 가져옴
        results = combine_batches(results, batch, file_format)   # 각 batch 결과를 누적 결과에 병합
        print_progress_batches(i, size, total)
    if file_format == "xml":
        return merge_xml_results(results)   # XML 포맷이면 여러 batch 결과를 하나의 XML로 병합
    return results


def get_id_mapping_results_stream(url):

    # 이 함수는 “streaming endpoint” 를 이용해 UniProt 매핑 결과를 한 번의 요청으로 전체 다운로드할 때 사용됩니다.
    # 즉, 여러 batch로 나눠 받지 않고, 한 번에 전체 결과를 스트리밍으로 다운로드하는 단축 버전

    if "/stream/" not in url:   # URL에 "/stream/"이 없으면, "/results/"를 "/results/stream/"으로 바꿔서 스트리밍 엔드포인트로 변경
        # 일반적으로 UniProt의 ID 매핑 결과 URL은 두 가지로 나뉨
        # /results/ → batch 기반, 여러 페이지로 나눠서 결과를 전송.
        # /results/stream/ → 전체 결과를 한 번에 스트리밍.
        url = url.replace("/results/", "/results/stream/")
    request = session.get(url)
    check_response(request)
    parsed = urlparse(url)   # URL을 구조적으로 분해
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)



def _make_sequence_file(ids, result, output_fname):
    eus = []
    for r in result["results"]:
        ensembl_id = r.get("from")   # 'from' 필드는 매핑의 원본 ID (즉, Ensembl ID)
        if ensembl_id in ids:
            ids.remove(ensembl_id)
            to_field = r.get("to", {})
            if "reviewed" in r["to"]["entryType"]:
                uniprot_id = to_field.get("primaryAccession")   # 인증된 Swiss-Prot ID(Uniprot id)
                if uniprot_id and not uniprot_id.startswith("A"):
                    sequence = to_field.get("sequence", {}).get("value")   # Uniprot id가 있을 경우 Uniprot 단백질 서열
                    if sequence:
                        eus.append([ensembl_id, uniprot_id, sequence])
                    else:
                        eus.append([ensembl_id, uniprot_id, "No_sequence"])
                else:
                    eus.append([ensembl_id, "Not_reviewed", None])

    for fail in result["failedIds"]:   # 'failedIds' 필드는 매핑에 실패한 ID 목록
        if fail in ids:
            ids.remove(fail)
            eus.append([fail, "No_Uniprot_Id", None])
        else:
            pass
        
    eus = sorted(eus, key=lambda x: x[0])
    print(len(eus))

    with open(output_fname, "w", encoding = "utf-8") as file:
        for Ens, Uni, Seq in eus:
            file.write(f"{Ens} {Uni} {Seq}\n")
            


def uniprot_request():
    input_fname = "biomedical/union_protein.txt"
    output_fname = "biomedical/HI_union_Uniprot_Sequence2.txt"

    ids = []
    with open(input_fname, encoding="utf-8") as f:
        for line in f:
            ids.append(line.strip())

    print(len(ids))

    job_id = submit_id_mapping(
        from_db="Ensembl", to_db="UniProtKB", ids=ids
    )
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        result = get_id_mapping_results_search(link)
        # Equivalently using the stream endpoint which is more demanding
        # on the API and so is less stable:
        # results = get_id_mapping_results_stream(link)
    _make_sequence_file(ids, result, output_fname)


