import pandas as pd
import psycopg2
import re
import copy
import psycopg2.extras as extras
import numpy as np
import os
import copy
from configparser import ConfigParser

# 메인 모듈 시작

# db_config.ini 유무 확인
parser = ConfigParser(converters={'list': lambda x: [i.strip().strip("\'") for i in x[1:-1].split(',')]})
if os.path.isfile('./config/db_config.ini'):
    parser.read('./config/db_config.ini')
else:
    raise Exception("change_config.py 실행을 통해 configuration 파일 먼저 작성해 주세요!")
print("===========데이터베이스 접속 정보 확인중===========")
print(parser.get('db', 'user'))
print(parser.get('ulcer_extract', 'query'))

##### DB 연결 #####
db = psycopg2.connect(
    host=parser.get('db', 'host'),
    dbname=parser.get('db', 'schema_name'),
    user=parser.get('db', 'user'),
    password=parser.get('db', 'pwd'),
    port=int(parser.get('db', 'port'))
)

cursor = db.cursor()

cursor.execute(
    parser.get('ulcer_extract', 'query')
)
row = cursor.fetchall()
len(f"Extracted row length: {row}")

row_copied = copy.deepcopy(row)

col_name = [desc[0] for desc in cursor.description]
ulcer_csv = pd.DataFrame(row_copied, columns=col_name)


########################################
# ------------------------------- 데이터 제거 -------------------------------
########################################
print("찾고자하는 조건과 다른 ulcer 포함 데이터 삭제 시작")

# ------------------------------- 1차 삭제 -------------------------------
# re.search를 이용하여 매칭되어야 할 단어들을 note_text에서 찾음
# flags = re.I 는 대소문자를 구분하지 않음 -> 대소문자는 달라도 매칭 됨
# 삭제해야 할 단어 전부 직접 작성, typo까지 고려함

delete_idx = []
def check_delete_idx(row):
    global delete_idx
    note_text = row['note_text']
    if re.search(r'ulcerofungat|ulceroinfiltrat', note_text, flags=re.I):
        delete_idx.append(row.name)
        return
    if re.search(r'resolved ileal ulcer', note_text, flags=re.I):
        delete_idx.append(row.name)
        return
    if re.search(r'prev ulcer|previous ulcer', note_text, flags=re.I):
        delete_idx.append(row.name)
        return
    if re.search(r'anastomo', note_text, flags=re.I):
        delete_idx.append(row.name)
        return
    if re.search(
            r'Digital rectal exam|Digital rectal exma|Digital examination of colostomy|Digital rectal exem|Digital colostomy exam|Digital  rectal exam|Digital ractal exam|Digital rectale exam|Digital recatal exam',
            note_text, flags=re.I):
        delete_idx.append(row.name)
        return
    if re.search(r'\bDRE(\b|시|상|)', note_text, flags=re.I):
        delete_idx.append(row.name)
        return

    if re.search('\r\n', note_text):
        sent_spt = note_text.split('\r\n\r\n')
    else:
        sent_spt = note_text.split('\n\n')
    for s in sent_spt:
        if re.search(r'\besd\b|\bemr\b', s, flags=re.I):
            if re.search(r'\besd\b|\bemr\b', s, flags=re.I) and re.search(r'ulcer', s, flags=re.I):
                # print(note_text)
                # print(s)
                delete_idx.append(row.name)
                return


delete_idx = list(set(delete_idx))
ulcer_csv.apply(check_delete_idx, axis=1)

print(f"기존 len: {len(ulcer_csv)}")
ulcer_csv.drop(delete_idx, inplace=True)
ulcer_csv.reset_index(drop=True, inplace=True)
print(f"1차 삭제후 len: {len(ulcer_csv)}")


# ------------------------------- 2차 삭제 -------------------------------
# 단어가 완전히 일치해야만 매칭됨, but 대소문자는 달라도 매칭됨
# 제거해야할 단어들은 concept_name_count에 나열되어 있음(행 수가 많아서 직접 치기 어려움)

df = pd.read_excel('concept_name_count.xlsx') 

# `df.regex`의 각 행을 '|' 기준으로 분리하고 결과를 리스트로 변환
split_data = df['regex'].str.split('|').explode().tolist()

df = pd.DataFrame(split_data, columns=['regex'])

j = 0
comma_idx = []
# ","가 포함된 여러개의 단어로 된 concept는 exact 매치 외에, 같은 paragraph내에 단어 모두 포함되면 다 삭제
# 따라서 "," 포함 concept, 미포함 concept 나눠서 진행 (미포함 먼저)
for i in range(len(df)):
    value = df['regex'][i]
    if isinstance(value, str) and re.search(",", df['regex'][i], flags=re.I):
        comma_idx.append(i)

df.drop(comma_idx, axis=0, inplace=True)


# "," 미포함인 concept와 full matching 함수
def concept_to_delete_text_check(name):
    matched_indices = []
    if not isinstance(name, str):
        return matched_indices
    
    name = re.escape(name)
    for i in range(len(ulcer_csv)):
        text = ulcer_csv.iloc[i]['note_text'].replace('\r\n', '')
        
        if re.search(name, text, flags=re.I):
            matched_indices.append(i)
            
    return matched_indices


all_matched_indices = []
for name_value in df['regex']:
    matched_indices = concept_to_delete_text_check(name_value)
    all_matched_indices.extend(matched_indices)

all_matched_indices = list(set(all_matched_indices))

print(f"기존 len: {len(ulcer_csv)}")
ulcer_csv = ulcer_csv.drop(all_matched_indices).reset_index(drop=True)
print(f"2차삭제 ','미포함 concept 삭제후 len: {len(ulcer_csv)}")


# "," 포함 concept 삭제 시작
df2 = pd.read_excel('concept_name_count.xlsx')

split_data = df2['regex'].str.split('|').explode().tolist()

df2 = pd.DataFrame(split_data, columns=['regex'])

new_idx = df2.index.difference(comma_idx)

df2 = df2.drop(new_idx)


def comma_split(row):
    spt = row['regex'].split(',')
    if len(spt) > 2:
        print(row)
        return
    return r'\b'+spt[0].strip()+r'\b', r'\b'+spt[1].strip()+r'\b'


df2[['reg1', 'reg2']] = df2.apply(comma_split, axis=1, result_type='expand')


# ------------------------------- E/S/D matching -------------------------------
# 단순 개행 혹은 개행*2로 문장 단위 나누는 경우 제대로 나뉘지 않음
# 따라서 내시경기록지의 구조인 Eso, Sto, Duo, Imp 4부분을 각각 나눠서 각 part 내부에서 단어 매칭 진행
# 먼저 네부분 text 따로 추출하기
not_matched_index = []
no_esd_index = []
wrong_index = []

# 각 부분의 시작이 다양한 단어로 나와있어 Eso, E, Esophagus, 식도와 같은 단어들을 모두 매칭시킴
print("E/S/D/Imp matching 시작")
print("시간이 조금 오래 걸립니다... (1시간 넘게 소요)")
def e_s_d_matching(row):
    global not_matched_index
    global no_esd_index
    global wrong_index
    if int(row.name) % 1000 == 0:
        print(row.name)
    regex_dict = {
        "Eso": r'(?:\b(?:Eso|E|Esophagus|식도)\W)(?P<Eso>(?:.|\n|\r\n)*)',
        "Sto": r'(?:\b(?:Sto|S|Stomach|G|위)\W)(?P<Sto>(?:.|\n|\r\n)*)',
        "Duo": r'(?:\b(?:Duo|D|Duodenum|십이지장)\W)(?P<Duo>(?:.|\n|\r\n)*)',
        "Imp": r'(?:(?:\b(?:Imp|Im|Impression|mp|lmp|conclusion|Imrpession|Ass)\W)|(?:\bP[\.\,\:\;\>\)]))(?P<Imp>(?:.|\n|\r\n)*)'
    }
    part_exist_list = []
    if re.search(regex_dict["Eso"], row['note_text']):
        part_exist_list.append("Eso")
    if re.search(regex_dict["Sto"], row['note_text']):
        part_exist_list.append("Sto")
    if re.search(regex_dict["Duo"], row['note_text']):
        part_exist_list.append("Duo")
    if re.search(regex_dict["Imp"], row['note_text']):
        part_exist_list.append("Imp")
    if len(part_exist_list)==0:
        not_matched_index.append(row.name)
        print("not matched")
        print(row.name)
        print("==========")
        return [np.nan, np.nan, np.nan, np.nan]
    elif part_exist_list[0]=="Imp":
        no_esd_index.append(row.name)
        print("no esd")
        print(row.name)
        print("==========")
        return [np.nan, np.nan, np.nan, np.nan]
    else:
        final_regex_string = r''
        final_out = [np.nan, np.nan, np.nan, np.nan]
        try:
            for p in part_exist_list:
                final_regex_string += regex_dict[p]
            m = re.search(final_regex_string, row['note_text'])
            for idx, (k,v) in enumerate(regex_dict.items()):
                if k in part_exist_list:
                    final_out[idx] = m.group(k)
            return final_out
        except AttributeError:
            print("wrong index")
            print(row.name)
            print("==========")
            wrong_index.append(row.name)
            return [np.nan, np.nan, np.nan, np.nan]


ulcer_csv[['Eso', 'Sto', 'Duo', 'Imp']] = ulcer_csv.apply(e_s_d_matching, axis=1, result_type='expand')

print(f"E/S/D/Imp 하나도 매칭되지 않은 케이스 개수: {len(not_matched_index)}")
print(f"Index: {not_matched_index}")
print(f"Imp만 매칭되는 케이스 개수: {len(no_esd_index)}")
print(f"Index: {no_esd_index}")
print(f"정규식과 맞지 않는 케이스 (해당 케이스 삭제): {len(wrong_index)}")
print(f"Index: {wrong_index}")

print(f"기존 len: {len(ulcer_csv)}")
ulcer_csv = ulcer_csv.drop(wrong_index).reset_index(drop=True)
print(f"정규식 매칭 오류 케이스 삭제후 len: {len(ulcer_csv)}")


# ulcer_split 파일로 저장해두기
# E/S/D/Imp 나누는게 정규식 매칭이 시간이 오래걸릴 수 있기 때문에 저장해놓고 사용
ulcer_csv.to_pickle('ulcer_split.pkl')

# 다시 불러와서 사용할 때 아래 주석 지우고 윗부분 돌리지 않고 아래만 쭉 실행
# ulcer_csv = pd.read_pickle('ulcer_split.pkl')

# 삭제 concept중 2개 단어 이상으로 되는 단어 실제 지우는 부분, Eso/Sto/Duo/Imp 각 부분안에서 matching 진행

delete_idx = []
for idx, row in ulcer_csv.iterrows():
    for inner_idx, inner_row in df2.iterrows():
        if not pd.isna(row['Eso']) and re.search(inner_row['reg1'], row['Eso'], flags=re.I) and re.search(inner_row['reg2'], row['Eso'], flags=re.I):
            delete_idx.append(idx)
            break
        elif not pd.isna(row['Sto']) and re.search(inner_row['reg1'], row['Sto'], flags=re.I) and re.search(inner_row['reg2'], row['Sto'], flags=re.I):
            delete_idx.append(idx)
            break
        elif not pd.isna(row['Duo']) and re.search(inner_row['reg1'], row['Duo'], flags=re.I) and re.search(inner_row['reg2'], row['Duo'], flags=re.I):
            delete_idx.append(idx)
            break
        elif not pd.isna(row['Imp']) and re.search(inner_row['reg1'], row['Imp'], flags=re.I) and re.search(inner_row['reg2'], row['Imp'], flags=re.I):
            delete_idx.append(idx)
            break

ulcer_csv.drop(delete_idx, axis=0, inplace=True)
ulcer_csv.reset_index(drop=True, inplace=True)

# 삭제케이스 모두 삭제 완료
# ------------------------------- 매칭 시작 -------------------------------
# 사용하는 concept인 BGU, GU, DU 매칭
print("=======================Concept matching 시작=======================")
# BGU 매칭
# concept_id와 concept_name 따로 작성
# bgu_index는 return 값을 concept_id로 받아서 넣음
ulcer_csv['concept_id'] = np.nan
ulcer_csv['concept_name'] = np.nan
bgu = []


def bgu_index(row):
    global bgu
    if re.search('benign gastric ulcer', row['note_text'], flags=re.I):
        bgu.append(row.name)
        return "4265600"
    if not pd.isna(row['Imp']) and re.search(r'\bBGU\b', row['Imp'], flags=re.I):
        bgu.append(row.name)
        return "4265600"
    if (not pd.isna(row['Imp']) and re.search(r'궤양', row['Imp'], flags=re.I)) and (not pd.isna(row['Sto']) and re.search(r'ulcer', row['Sto'], flags=re.I)):
        bgu.append(row.name)
        return "4265600"
ulcer_csv['concept_id'] = ulcer_csv.apply(bgu_index, axis=1)

def bgu_index_name(row):
    global bgu
    if re.search('benign gastric ulcer', row['note_text'], flags=re.I):
        return "Benign gastric ulcer"
    if not pd.isna(row['Imp']) and re.search(r'\bBGU\b', row['Imp'], flags=re.I):
        return "Benign gastric ulcer"
    if (not pd.isna(row['Imp']) and re.search(r'궤양', row['Imp'], flags=re.I)) and (not pd.isna(row['Sto']) and re.search(r'ulcer', row['Sto'], flags=re.I)):
        return "Benign gastric ulcer"
ulcer_csv['concept_name'] = ulcer_csv.apply(bgu_index_name, axis=1)

# 잘 들어갔는지 확인
len(f"BGU count: {len(bgu)}")
print(ulcer_csv['concept_id'].value_counts())
print()
print(ulcer_csv['concept_name'].value_counts())


# GU 매칭
# pd.notna 확인 후 그대로 return 하는 조건 추가
# 즉, BGU 이미 매칭된 case면 GU matching 안함
gu = []

def gu_index(row):
    global gu
    if pd.notna(row['concept_id']):
        return row['concept_id']
    
    if re.search('gastric ulcer', row['note_text'], flags=re.I):
        gu.append(row.name)
        return "4265600"
    if not pd.isna(row['Imp']) and re.search(r'\bGU\b', row['Imp'], flags=re.I):
        gu.append(row.name)
        return "4265600"
    if (not pd.isna(row['Imp']) and re.search(r'궤양', row['Imp'], flags=re.I)) and (not pd.isna(row['Sto']) and re.search(r'ulcer', row['Sto'], flags=re.I)):
        gu.append(row.name)
        return "4265600"
ulcer_csv['concept_id'] = ulcer_csv.apply(gu_index, axis=1)

def gu_index_name(row):
    global gu
    if pd.notna(row['concept_name']):
        return row['concept_name']
    
    if re.search('gastric ulcer', row['note_text'], flags=re.I):
        return "Gastric ulcer"
    if not pd.isna(row['Imp']) and re.search(r'\bGU\b', row['Imp'], flags=re.I):
        return "Gastric ulcer"
    if (not pd.isna(row['Imp']) and re.search(r'궤양', row['Imp'], flags=re.I)) and (not pd.isna(row['Sto']) and re.search(r'ulcer', row['Sto'], flags=re.I)):
        return "Gastric ulcer"
ulcer_csv['concept_name'] = ulcer_csv.apply(gu_index_name, axis=1)

# GU 넣고 확인
len(f"GU count: {len(gu)}")
print(ulcer_csv['concept_id'].value_counts())
print()
print(ulcer_csv['concept_name'].value_counts())


# DU 매칭
du = []
# DU의 경우 BGU/GU 합집합과 같이 matching 되는 경우 concept 둘다 추가하기 위해 동시 매칭 따로 빼서 진행
# 우선 matching index만 뽑고 concept_id, name은 나중에 추가
def du_index(row):
    global du
    if re.search('duodenal ulcer', row['note_text'], flags=re.I):
        du.append(row.name)
        return #"4198381"
    if not pd.isna(row['Imp']) and re.search(r'\bDU\b', row['Imp'], flags=re.I):
        du.append(row.name)
        return #"4198381"
    if (not pd.isna(row['Imp']) and re.search(r'궤양', row['Imp'], flags=re.I)) and (not pd.isna(row['Duo']) and re.search(r'ulcer', row['Duo'], flags=re.I)):
        du.append(row.name)
        return #"4198381"

ulcer_csv.apply(du_index, axis=1)

# 동시에 매칭되는 부분 확인
bgu_set = set(bgu)
gu_set = set(gu)
du_set = set(du)
bgu_gu_set = bgu_set.intersection(gu_set)
print(f"BGU&GU 동시 매칭: {len(bgu_gu_set)}")
gu_du_set = gu_set.intersection(du_set)
print(f"GU&DU 동시 매칭: {len(gu_du_set)}")
bgu_du_set = bgu_set.intersection(du_set)
print(f"BGU&DU 동시 매칭: {len(bgu_du_set)}")
bgu_gu_du_set = bgu_gu_set.intersection(du_set)
print(f"BGU&GU&DU 동시 매칭: {len(bgu_gu_du_set)}")

bgu_du_both_set = set((bgu_set|gu_set)&du_set)
print(f"(BGU|GU)&DU 동시 매칭: {len(bgu_du_both_set)}")

# DU only matching 먼저 concept_id 넣기
def du_index(row):
    if pd.notna(row['concept_id']):
        return row['concept_id']
    
    if re.search('duodenal ulcer', row['note_text'], flags=re.I):
        return "4198381"
    if not pd.isna(row['Imp']) and re.search(r'\bDU\b', row['Imp'], flags=re.I):
        return "4198381"
    if (not pd.isna(row['Imp']) and re.search(r'궤양', row['Imp'], flags=re.I)) and (not pd.isna(row['Duo']) and re.search(r'ulcer', row['Duo'], flags=re.I)):
        return "4198381"
ulcer_csv['concept_id'] = ulcer_csv.apply(du_index, axis=1)

# 미리 뽑아둔 DU와 BGU/DU 합집합 동시에 있는 case 새로운 행으로 추가후 DU concept까지 넣기
both_list = sorted(list(bgu_du_both_set))
add_df = pd.DataFrame(columns=ulcer_csv.columns)

for idx, row in ulcer_csv.iterrows():
    if idx in both_list:
        new_row = copy.deepcopy(row)
        new_row['concept_id'] = "4198381"
        add_df = add_df.append(new_row)

print(f"DU 케이스 추가전: {len(ulcer_csv)}")
ulcer_csv = pd.concat([ulcer_csv, add_df]).sort_index()
print(f"DU와 (BGU|GU) 동시 매칭 case DU concept id 추가후: {len(ulcer_csv)}")
def du_index_name(row):

    if row['concept_id'] == "4198381":
        return "Duodenal ulcer"
    else:
        return row['concept_name']
ulcer_csv['concept_name'] = ulcer_csv.apply(du_index_name, axis=1)

# 최종적으로 BGU or GU or DU 1개라도 매칭된 set
total = (bgu_set|gu_set|du_set)



# dieulafoy 매칭

dieulafoy = []
def dieulafoy_index(row):
    global dieulafoy
    
    if re.search('dieulafoy', row['note_text'], flags=re.I):
        dieulafoy.append(row.name)
        return "198798"
    else:
        return row['concept_id']
ulcer_csv['concept_id'] = ulcer_csv.apply(dieulafoy_index, axis=1)


def dieulafoy_index_name(row):
    if row['concept_id'] == "198798":
        return "Dieulafoy's ulcer"
    else:
        return row['concept_name']
ulcer_csv['concept_name'] = ulcer_csv.apply(dieulafoy_index_name, axis=1)


# dieulafoy 들어가고 확인
print(f"Dieulafoy 개수: {dieulafoy}")
print(ulcer_csv['concept_id'].value_counts())
print()
print(ulcer_csv['concept_name'].value_counts())

# 최종적으로 BGU/GU/DU/Dieulafoy 1개라도 매칭되지 않는 경우 모두 삭제
ulcer_csv.dropna(subset=['concept_id'], inplace=True)


# ------------------------------- Bleeding 매칭 시작 -------------------------------
# Bleeding 키워드 존재하고 negation도 존재하는 경우는 mild로 생각
# Bleeding 키워드 존재하고 negation term 없는 경우만 bleeding ulcer concept로 변환

bgu_bleeding = []
bgu_bleeding_neg = []

def bgu_bleeding_index(row):
    global bgu_bleeding
    global bgu_bleeding_neg
    if row['concept_id'] == "4265600":
        if (not pd.isna(row['Sto']) and re.search(r'active bleeding|Forrest|F {1,4}(Ia|Ib|IIa|IIb)', row['Sto'], flags=re.I)) or (not pd.isna(row['Imp']) and re.search(r'active bleeding|Forrest|F {1,4}(Ia|Ib|IIa|IIb)', row['Imp'], flags=re.I)):
            if not pd.isna(row['Sto']) and re.search(r'active bleeding|Forrest|(F {1,4}(Ia|Ib|IIa|IIb))', row['Sto'], flags=re.I) and re.search(r'\bno\b|않음|없음|없으|없었|없어|없는|없 음|않았', row['Sto'], flags=re.I):
                bgu_bleeding_neg.append(row.name)
                return row['concept_id']
            if not pd.isna(row['Sto']) and re.search(r'active bleeding|Forrest|(F {1,4}(Ia|Ib|IIa|IIb))', row['Sto'], flags=re.I) and re.search(r'가능성', row['Sto'], flags=re.I) and re.search(r'낮음|낮은', row['Sto'], flags=re.I):
                bgu_bleeding_neg.append(row.name)
                return row['concept_id']
            if not pd.isna(row['Imp']) and re.search(r'active bleeding', row['Imp'], flags=re.I) and re.search(r'\bno\b|않음|없음|없으|없었|없어|없는', row['Imp'], flags=re.I):
                bgu_bleeding_neg.append(row.name)
                return row['concept_id']
            bgu_bleeding.append(row.name)
            return "4231580"
    return row['concept_id']
ulcer_csv['concept_id'] = ulcer_csv.apply(bgu_bleeding_index, axis=1)

def bgu_bleeding_index_name(row):

    if row['concept_id'] == "4231580":
        return "Gastric ulcer, acute with hemorrhage"
    else:
        return row['concept_name']
ulcer_csv['concept_name'] = ulcer_csv.apply(bgu_bleeding_index_name, axis=1)

print(f"BGU_bleeding 전체 케이스: {len(bgu_bleeding)}")
print(f"BGU_bleeding with negation 케이스: {len(bgu_bleeding_neg)}")
print(f"최종 BGU_bleeding 케이스: {len(bgu_bleeding) - len(bgu_bleeding_neg)}")

du_bleeding = []
du_bleeding_neg = []
def du_bleeding_index(row):
    global du_bleeding
    global du_bleeding_neg
    if row['concept_id'] == "4198381":
        if (not pd.isna(row['Duo']) and re.search(r'active bleeding|Forrest|F {1,4}(Ia|Ib|IIa|IIb)', row['Duo'], flags=re.I)) or (not pd.isna(row['Imp']) and re.search(r'active bleeding|Forrest|F {1,4}(Ia|Ib|IIa|IIb)', row['Imp'], flags=re.I)):
            if not pd.isna(row['Duo']) and re.search(r'active bleeding|Forrest|F {1,4}(Ia|Ib|IIa|IIb)', row['Duo'], flags=re.I) and re.search(r'\bno\b|않음|없음|없으|없었|없어|없는|없 음|않았', row['Duo'], flags=re.I):
                du_bleeding_neg.append(row.name)
                return row['concept_id']
            if not pd.isna(row['Duo']) and re.search(r'active bleeding|Forrest|F {1,4}(Ia|Ib|IIa|IIb)', row['Duo'], flags=re.I) and re.search(r'가능성', row['Duo'], flags=re.I) and re.search(r'낮음|낮은', row['Duo'], flags=re.I):
                du_bleeding_neg.append(row.name)
                return row['concept_id']
            if not pd.isna(row['Imp']) and re.search(r'active bleeding', row['Imp'], flags=re.I) and re.search(r'\bno\b|않음|없음|없으|없었|없어|없는', row['Imp'], flags=re.I):
                du_bleeding_neg.append(row.name)
                return row['concept_id']
            du_bleeding.append(row.name)
            return "4027729"
    return row['concept_id']
ulcer_csv['concept_id'] = ulcer_csv.apply(du_bleeding_index, axis=1)

def duo_bleeding_index_name(row):
    if row['concept_id'] == "4027729":
        return "Acute duodenal ulcer with hemorrhage"
    else:
        return row['concept_name']
ulcer_csv['concept_name'] = ulcer_csv.apply(duo_bleeding_index_name, axis=1)

print(f"DU_bleeding 전체 케이스: {len(du_bleeding)}")
print(f"DU_bleeding with negation 케이스: {len(du_bleeding_neg)}")
print(f"최종 DU_bleeding 케이스: {len(du_bleeding) - len(du_bleeding_neg)}")
ulcer_csv.drop_duplicates(inplace=True)

# bleeding 들어가고 확인
print("최종 concept_id, concept_name count")
print(ulcer_csv['concept_id'].value_counts())
print()
print(ulcer_csv['concept_name'].value_counts())
ulcer_csv.reset_index(drop=True, inplace=True)


# ------------------------------- 최종 매칭된 dataframe insert 시작 -------------------------------
## NULL string으로 만들고 아래에서 tuple로 만들때 None으로 다시 바꿔주기
## 그냥 None값 그대로 대입시 실제 INSERT 할 때 NaT string으로 들어가서 date type에 맞지 않아 오류 발생

ulcer_csv['concept_id'].astype(int)
ulcer_csv = ulcer_csv.fillna("NULL")

tups = [tuple(x) for x in ulcer_csv.to_numpy()]
new_tup_for_null = []

##### DB 와 연결 #####
## NULL값 None으로 변경
for t in tups:
    tmp = list(t)
    for i in range(len(t)):
        if t[i] == "NULL":
            tmp[i] = None
    new_tup_for_null.append(tuple(tmp))

cols = ', '.join(list(ulcer_csv.columns))

query = f"INSERT INTO {parser.get('ulcer_extract', 'out_schema')}.{parser.get('ulcer_extract', 'out_table')} ( {cols} ) VALUES %s;"
cursor = db.cursor()
try:
    extras.execute_values(cursor, query, new_tup_for_null)
except (Exception, psycopg2.DatabaseError) as error:
    print("Error: %s" % error)
    db.rollback()
    cursor.close()

db.commit()
cursor.close()
db.close()



