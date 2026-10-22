# 우리집에 깔아보기 — 3D 모델 빌드

`python3 build_models.py [green gray beige ivory]` → `../models/{color}_{size}.glb` 와 `../thumb_{color}.jpg` 를 다시 만듦 (색을 안 적으면 4색 전부).
필요: numpy, pillow, scipy.

- `SRC` = 색별 (원단 결 원본, 자를 영역, 그 영역의 실제 cm, 목표 base 색 sRGB, 결 진하기)
- 목표 색은 실제 방 영상/스튜디오 사진 속 러그 색을 AR 조명 감쇠만큼 살짝 올린 값
- `SIZES` = 사이즈 키·가로·세로(cm). index.html 의 SIZES 와 키를 맞출 것
- ivory 는 탑뷰 원본이 없어 beige 사진의 결 + 아이보리 연출컷에서 뽑은 색 (탑뷰 촬영본이 생기면 SRC['ivory'] 원본만 교체)

src/ 원본 (구글 드라이브 우리 촬영본):
- green_3141_6s.jpg — 그린 소재/동영상 「1021_IMG_3141 - 이다영.mov」 6초 프레임
- gray_close.jpg — 08) 펫슬리브/상세페이지ver1 「근접 연출_그레이.JPG」
- beige_d3070.jpg — 250110_펫슬리브_스튜디오촬영/아이폰_전체 「DSCF3070.JPG」
