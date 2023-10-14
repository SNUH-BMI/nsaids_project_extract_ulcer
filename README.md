<<<<<<< HEAD
# nsaids_project_extract_ulcer
=======
# nsaids_project_extract_ulcer

## CDM의 note table에 접근해 내시경기록지를 불러와 "ulcer"와 관련된 text를 추출하는 코드입니다.

=================== 실행방법 ===================

1. requirements.txt에 나온 python library를 모두 설치해주세요.
2. 프로젝트 폴더안에 "config"라는 이름의 폴더를 생성해주세요.
3. ini파일 생성을 위해 change_config.py에 db관련 정보를 알맞게 입력해주세요.
* 'db'
   * "host": CDM db 접속 ip 주소를 입력해주세요.
   * "port": CDM db 접속 port를 입력해주세요.
   * "user": CDM db 접속 id를 입력해주세요.
   * "pwd": user 계정의 password 주소를 입력해주세요.
   * "schema_name": 사용할 스키마 이름을 입력해주세요.
* 'ulcer_extract'
   * "query": "ulcer" 텍스트가 포함된 모든 내시경기록지 추출이 가능한 query를 입력해주세요.
   * "out_schema": 최종 추출된 데이터를 저장할 스키마 이름을 입력해주세요.
   * "out_table": 최종 추출된 데이터를 저장할 테이블 이름을 입력해주세요.
   <br>
* 예시
   <br>
![image](https://github.com/SNUH-BMI/nsaids_project_extract_ulcer/assets/19829142/90301426-cd1a-4b63-84b4-85d08b96d7d6)

4. change_config.py를 실행해주세요. (예. python change_config.py)
5. 접속한 database에 out_schema.out_table 이름의 테이블을 생성해주세요.
* out_table의 기본구조는 CDM의 note 테이블과 동일한 컬럼들을 가지고 있습니다.
   * note 테이블과 동일한 구조의 out_schema.out_table 테이블을 생성해주세요.
   * 예시). CREATE TABLE out_schema.out_table (like schema_name.note)
* 추가로 Eso(type=text), Sto(type=text), Duo(type=text), Imp(type=text), concept_id(type=bigint), concept_name(type=varchar(50)) 형식의 column들을 추가해주세요.
   * 각 note에서 파싱된 Eso,Sto,Duo,Imp 부분 및 붙여진 concept_id, concept_name이 추출될 컬럼들입니다.

<br>
6. 준비가 모두 완료되었습니다. nsaids_project.py 파일을 실행해주세요.
<br>
※ E/S/D/Imp matching 시작 부분이 조금 오래걸릴 수 있습니다.
>>>>>>> 4876fa1c9671f0c8db8b6920f69691a2ad46fc9b
