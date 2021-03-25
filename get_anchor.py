import math
lane_num = 18

for i in range(1, lane_num + 1):
    anchors = (590-(i-1)*20)-1
    anchors = math.floor((256 / 590) * anchors)
    print(anchors)