# 우리집에 깔아보기 — 3D 모델 빌드 (v3)

`python3 build_models.py [green gray beige ivory]` → `../models/{color}_{size}.glb`, `../models/{color}_{size}_r90.glb`, `../thumb_{color}.jpg` 를 다시 만듦 (색을 안 적으면 4색 전부, 32×2 파일, 약 9MB, 10초).
필요: numpy, pillow, scipy.

## 만드는 방식
- 상판: 둥근 모서리(r 3cm) 사각형, 옆면은 1/4 원으로 둥글린 파이핑 테두리(두께 9mm), 바닥은 짙은 회갈색 논슬립 면.
- 원단 결: 사진에서 2cm 이상 명암(조명·주름)을 빼고 결만 남긴 뒤, 결 방향을 자동으로 찾아 러그 긴 변에 맞춰 돌림.
  결 방향으로 길쭉한 깃털 조각을 랜덤으로 겹쳐 붙여서 헤더 줄무늬가 끊기지 않고 반복 무늬도 없음.
  결 진하기는 사진 실측 대비 그대로(그린 0.09, 나머지는 사진이 흐려 최소 0.02).
- 가장자리 1.2cm 는 파이핑(둥근 면, 결이 짧게 끊김) + 박음질 홈 + 땀 자국.
- 노멀맵: 파이핑·홈·완만한 굴곡만. 미세 보풀 노이즈는 안 넣음(JPEG 압축 잡티 방지). 해상도는 알베도의 절반.
- 텍스처는 2의 거듭제곱 크기(밉맵용), 긴 변 최대 2048. 알베도 JPEG q90, 노멀 JPEG q92.
- `_r90` 판: 같은 모델에 Y축 90° 회전만 심은 것. 안드로이드 Scene Viewer(WebXR 없는 기기)용.
  아이폰·WebXR 은 페이지에서 GLB 의 노드 회전을 직접 고쳐 blob 으로 넘기므로 15° 단위 회전이 AR 에 그대로 반영됨.

## 설정
- `SRC` = 색별 (결 원본 사진, 자를 영역, 그 영역의 실제 가로 cm, 목표 base 색 sRGB)
- 목표 색은 실제 방 영상/스튜디오 사진 속 러그 색을 AR 조명 감쇠만큼 살짝 올린 값
  - green (112,152,149) · gray (178,174,171) · beige (204,194,184) · ivory (224,218,207)
- `SIZES` = 사이즈 키·가로·세로(cm). index.html 의 SIZES 와 키를 맞출 것

## src/ 원본 (전부 우리 촬영본, 구글 드라이브)
- green_3141_6s.jpg — 「1021_IMG_3141 - 이다영.mov」 6초 프레임 (1RNeeu0xCk8OFE0fJnUzgZ1-d25Lh8dQs)
- gray_close.jpg — 08) 펫슬리브/상세페이지ver1 「근접 연출_그레이.JPG」 (1HsklTH4r1-Sz3QU68R6T9pnwPKVVDYrS)
- beige_d3070.jpg — 250110 스튜디오촬영 「DSCF3070.JPG」 (1ShkWieUhi1-FDFo21xyhLpIVqIxsES2B)
- ivory_top_250625.jpg — 「250625펫슬리브아이보리거실XL_황호준.jpg」(1HwPAfOe0_KGfUPc4Gg4OExcnVfwqYcqw) 가운데 잘라낸 탑뷰. 플래시로 밝게 날아갔지만 결은 또렷해 결 원본으로만 쓰고 색은 목표값 사용.

## 아임웹 상세페이지에 넣기
버튼 하나로 링크 (추천 — iframe 안에서는 iOS Quick Look 이 안 열릴 수 있음):
```html
<a href="https://yourdogloves.github.io/reliv-guides/ar/?size=XL&color=green" target="_blank" rel="noopener"
   style="display:block;text-align:center;padding:14px;border-radius:12px;background:#2B271C;color:#fff;font-weight:700;text-decoration:none">
  📷 우리집 바닥에 깔아보기 (AR)
</a>
```
옵션별로 `?size=S|M|L|XL|hall|widehall|module|kitchen&color=green|gray|beige|ivory`, 미리 돌려둘 땐 `&rot=90`.
헤더·사용법 없이 미리보기만 보여주고 싶으면 `&embed=1`.
