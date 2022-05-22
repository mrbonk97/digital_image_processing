import cv2

img1 = cv2.imread('picuture.jpg', 0)
img2 = cv2.imread('test3.jpg', 0)

img1 = cv2.resize(img1, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)



surf = cv2.xfeatures2d.SURF_create(10000)

kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)


matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# 매칭 계산 ---④
matches = matcher.match(des1, des2)
# 매칭 결과 그리기 ---⑤
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# res = cv2.resize(res, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
cv2.imshow('BF + SURF', res)
cv2.waitKey()
cv2.destroyAllWindows()