import cv2
import numpy as np
import matplotlib.pyplot as plt


def warp_image(image):

    img = image.copy()
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    temp = np.zeros_like(gray_img[:, :])
    src_pts = np.array([[225, 670], [545, 450], [780, 450], [1200, 670]])
    cv2.fillConvexPoly(temp, src_pts, (255, 255, 255))
    binary_img = cv2.bitwise_and(gray_img, gray_img, mask=temp)

    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(8,8))
    binary_img = clahe.apply(binary_img)
    binary_img = cv2.medianBlur(binary_img, 15)
    _, binary_img = cv2.threshold(binary_img, 170, 255, cv2.THRESH_BINARY)
    
    dst_pts = np.array([[0, 720], [0, 0], [1280, 0], [1280, 720]])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, H, (1280, 720))

    binary_warped = cv2.cvtColor(warped_img.copy(), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(8,8))
    binary_warped = clahe.apply(binary_warped)
    binary_warped = cv2.medianBlur(binary_warped, 15)
    _, binary_warped = cv2.threshold(binary_warped, 170, 255, cv2.THRESH_BINARY)

    return binary_img, binary_warped, H


def detect_lane(image, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzero):

    org_img = image.copy()

    binary_img, binary_warped, H = warp_image(image)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    binary_disp = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)

    lane_found = False

    if (10000 < (np.sum(binary_warped == 255)) < 40000):

        lane_found = True

        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 10
        window_height = round(binary_warped.shape[0]/nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        width = 80
        minpix = 40
        left_lane_inds = []
        right_lane_inds = []
        rectangle_data = []

        for window in range(nwindows):

            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - width
            win_xleft_high = leftx_current + width
            win_xright_low = rightx_current - width
            win_xright_high = rightx_current + width
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

            cv2.ellipse(binary_disp, (round((win_xleft_low + win_xleft_high)/2), round((win_y_low + win_y_high)/2)), (round(abs(win_xleft_high - win_xleft_low)/2), round(abs(win_y_high - win_y_low)/2)), 0, 0, 360, (0,255,0), 2)
            cv2.ellipse(binary_disp, (round((win_xright_low + win_xright_high)/2), round((win_y_low + win_y_high)/2)), (round(abs(win_xright_high - win_xright_low)/2), round(abs(win_y_high - win_y_low)/2)), 0, 0, 360, (0,255,0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    binary_disp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    binary_disp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

    h = binary_warped.shape[0]
    ym_per_pix = 3.048/100
    xm_per_pix = 3.7/378
    y_eval = np.max(ploty)

    if len(leftx) != 0 and len(rightx) != 0:

        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])  

    left_plot_lines = (np.hstack([left_fit_x[:, np.newaxis], ploty[:, np.newaxis]])).reshape((-1, 1, 2))
    right_plot_lines = (np.hstack([right_fit_x[:, np.newaxis], ploty[:, np.newaxis]])).reshape((-1, 1, 2))

    cv2.polylines(binary_disp, np.int32([left_plot_lines]), False, (0 ,255, 255), 2)
    cv2.polylines(binary_disp, np.int32([right_plot_lines]), False, (0 ,255, 255), 2)

    warp_left_coord = np.vstack((left_fit_x, ploty, np.ones(left_fit_x.shape)))
    warp_right_coord = np.vstack((right_fit_x, ploty, np.ones(right_fit_x.shape)))

    left_org_coord = np.matmul(np.linalg.inv(H), warp_left_coord)
    left_org_coord = np.divide(left_org_coord, left_org_coord[-1]).astype(int)

    right_org_coord = np.matmul(np.linalg.inv(H), warp_right_coord)
    right_org_coord = np.divide(right_org_coord, right_org_coord[-1]).astype(int)

    left_org_plot_lines = (np.hstack([left_org_coord[0, :, np.newaxis], left_org_coord[1, :, np.newaxis]])).reshape((-1, 1, 2))
    right_org_plot_lines = (np.hstack([right_org_coord[0, :, np.newaxis], right_org_coord[1, :, np.newaxis]])).reshape((-1, 1, 2))

    pts = np.array([[left_org_coord[0, np.argmax(left_org_coord[1])], left_org_coord[1, np.argmax(left_org_coord[1])]], [left_org_coord[0, np.argmin(left_org_coord[1])], left_org_coord[1, np.argmin(left_org_coord[1])]], [right_org_coord[0, np.argmin(right_org_coord[1])], right_org_coord[1, np.argmin(right_org_coord[1])]], [right_org_coord[0, np.argmax(right_org_coord[1])], right_org_coord[1, np.argmax(right_org_coord[1])]]])

    mask = np.zeros_like(image[:, :])
    cv2.fillPoly(mask, np.int32([pts]), color=(0, 255, 0))
    image = cv2.addWeighted(image, 1, mask, 0.5, 0)

    cv2.polylines(image, np.int32([left_org_plot_lines]), False, (0 , 0, 255), 4)
    cv2.polylines(image, np.int32([right_org_plot_lines]), False, (255 , 0, 0), 4)

    org_img = cv2.resize(org_img, (320, 180), interpolation = cv2.INTER_AREA)
    binary_img = cv2.resize(binary_img, (320, 180), interpolation = cv2.INTER_AREA)
    binary_disp = cv2.resize(binary_disp, (320, 180), interpolation = cv2.INTER_AREA)
    binary_disp = np.concatenate((org_img, binary_img, binary_disp), axis=0)

    image = cv2.resize(image, (960, 540), interpolation = cv2.INTER_AREA)
    empty2 = np.zeros((180, 1280, 3), np.uint8)

    vis = np.concatenate((image, binary_disp), axis=1)
    vis = np.concatenate((vis, empty2), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Left Curvature: ' + '{:04.2f}'.format(left_curverad) + 'm' + ', ' + 'Right Curvature: ' + '{:04.2f}'.format(right_curverad) + 'm'
    cv2.putText(vis, text, (40,600), font, 0.75, (255,255,255), 2, cv2.LINE_AA)

    text = 'Average Curvature: ' + '{:04.2f}'.format(((left_curverad + right_curverad)/2)) + 'm'
    cv2.putText(vis, text, (40,630), font, 0.75, (255,255,255), 2, cv2.LINE_AA)

    if lane_found:
        text = 'Lane found'
        cv2.putText(vis, text, (40,660), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        text = 'Lane not found'
        cv2.putText(vis, text, (40,660), font, 0.75, (60,20,220), 2, cv2.LINE_AA)

    if ((((left_curverad + right_curverad)/2)) > 200):
        text = 'TURN RIGHT --->'
    elif ((((left_curverad + right_curverad)/2)) < 100):
        text = 'TURN LEFT <---'
    else:
        text = 'GO STRAIGHT'
    cv2.putText(vis, text, (40,690), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    return vis, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzero


def predict_turn(video):

    cap = cv2.VideoCapture(video)
    # out = cv2.VideoWriter('../output/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

    left_fit, right_fit = (None, None)
    left_lane_inds, right_lane_inds = (None, None)
    nonzero = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        vis, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzero = detect_lane(frame, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzero)

        # out.write(vis)
            
        cv2.imshow('Problem3 Output', vis)
        cv2.waitKey(25)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    predict_turn('../data/challenge.mp4')