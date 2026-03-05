import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import time


st.title("Hello Streamlit 👋")
st.write("이것은 첫 번째 스트림릿 앱입니다.")

name = st.text_input("이름을 입력하세요")
if st.button("인사하기"):
    st.success(f"안녕하세요, {name}님!")

st.title("스트림릿 제목")
st.header("헤더")
st.subheader("서브헤더")
st.text("일반 텍스트")
st.markdown("**마크다운 지원** :sparkles:")
st.code("print('Hello World')", language="python")

col1, col2 = st.columns(2)
col1.write("왼쪽 컬럼")
st.write("이것은 왼쪽 컬럼입니다.")
col2.write("오른쪽 컬럼")
st.write("이것은 오른쪽 컬럼입니다.")

with st.expander("펼치기/접기"):
    st.write("숨겨진 내용")

name = st.text_input("이름 입력")

age = st.number_input("나이 입력", min_value=0, max_value=120, value=25)
score = st.slider("점수", 0, 100, 50)

agree = st.checkbox("동의합니다")
disagree = st.checkbox("동의하지않습니다")
option = st.radio("좋아하는 색상", ["빨강", "파랑", "초록"])
select = st.selectbox("과목 선택", ["수학", "과학", "영어"])
multi = st.multiselect("취미 선택", ["독서", "운동", "게임"])

if st.button("클릭"):
    st.success("버튼 눌림")

uploaded_file = st.file_uploader("파일 업로드", type=["jpg","png","csv"])
if st.button("이미지 확인"):
    if uploaded_file == None:
        st.error("이미지를 업로드 하세요.")
    else:
        st.image(uploaded_file, caption="업로드", use_container_width=True)

img = Image.open("./data/celeb/고윤정/Image_36.jpg")
st.image(img, caption="고윤정", use_container_width=True)

# st.audio("music.mp3")
# st.video("video.mp4")

df = pd.DataFrame({"이름":["철수","영희"], "점수":[90,80]})
st.table(df)       # 정적 테이블
st.dataframe(df)   # 인터랙티브 테이블

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a","b","c"])
st.line_chart(chart_data)
st.bar_chart(chart_data)
st.area_chart(chart_data)

progress = st.progress(0)
for i in range(100):
    time.sleep(0.05)
    progress.progress(i+1)

st.success("성공 메시지")
st.error("에러 메시지")
st.warning("경고 메시지")
st.info("정보 메시지")


