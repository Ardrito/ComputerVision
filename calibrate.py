import cv2 as cv
import numpy as np
import glob

def getCorners(chessboardSize: tuple[int,int] = (10,7), frameSize: tuple[int,int]=(1280,720), 
               path: str = "images/", fileType: str = ".jpeg", showImages: bool = False, square_size:float =0.018):
    '''
    Find corners of squares in images of checkerboards from a folder of images
    
    Recommended between 5-10 images and non square checkerboard of at least 5x4, recommended 

    path as "path/to/file/"

    Returns array (image, points, location(x,y))
    '''
    assert (chessboardSize[0] > 4) & (chessboardSize[1] > 3) & (chessboardSize[0] > chessboardSize[1]), "Chessboard should be at least 5x4 and for an M x N, M>N"
    assert (frameSize[0] > 0) & (frameSize[1] > 0), "frame size should be non zero"

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #square_size = 0.018  # meters (18 mm)
    worldP = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    worldP[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    worldP *= square_size


    worldPoints = []
    imgPoints = []

    images = glob.glob(path+"*"+fileType)
    assert len(images) > 0, "Path should not be empty"

    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret == True:
            worldPoints.append(worldP)
            cornersSubPix = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgPoints.append(corners)

            if showImages == True:
                cv.drawChessboardCorners(img, chessboardSize, cornersSubPix, ret)
                cv.imshow("img", img)
                cv.waitKey(1000)

    cv.destroyAllWindows()

    return(worldPoints,imgPoints)

def getParams(worldPoints, imgPoints, frameSize):
    '''
    Get calibration parameters for a camera

    Saves a numpy file and returns values
    '''
    repError, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(worldPoints, imgPoints, frameSize,None,None)

    #print ("repError: " , repError)

    # np.savez(
    #     savePath,
    #     repError=repError,
    #     cameraMatrix=cameraMatrix,
    #     dist=dist,
    #     rvecs=rvecs,
    #     tvecs=tvecs
    # )

    return cameraMatrix, dist, rvecs, tvecs

def removeDistortion(cameraMatrix, dist, path: str = "images/", fileType: str = ".jpeg"):

    '''
    Removes distortion from an image and saves it in path
    '''
    img = cv.imread(glob.glob(path+"*"+fileType)[0])
    h,w = img.shape[:2]
    newCamMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix,dist,(w,h),1,(w,h))

    undistImg = cv.undistort(img, cameraMatrix, dist, None, newCamMatrix)
    x,y,h,w = roi

    undistImg = undistImg[y:y+h, x:x+w]

    cv.imwrite("undistorted.jpeg", undistImg)

    return (undistImg)

    # cv.imshow("img", img)
    # cv.imshow("undist ", undistImg)
    # cv.waitKey(10000)

def checkError(worldPoints, imgPoints, rvecs, tvecs, cameraMatrix, dist) -> float:
    '''
    Calculate reprojection error
    '''
    meanError = 0

    for i in range(len(worldPoints)):
        newImgPoints, _ = cv.projectPoints(worldPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgPoints[i], newImgPoints, cv.NORM_L2)/len(newImgPoints)
        meanError += error

    error = meanError/len(worldPoints)

    print ("Error is", error)

    return error


if __name__ == "__main__":
    # print("------Welcome------")
    # print("Enter values as prompted (leave blank for default values): ")
    # images_right = input("Path to images from right camera (From your point of view): ")
    # images_left = input("Path to images from left camera (From your point of view): ")
    # save_path = input("Save path or file name: ")
    # T = float(input("Distance between cameras in Meters: "))
    # square_size = float(input("Calibration squares size: "))
    # chessboard_horizontal = int(input("Number of horizontal calibration squares: "))
    # chessboard_vertical = int(input("Number of vertical calibration squares: "))
    # showImages = input("Show calibration images (yes/no): ")
    # R = np.eye(3)

    images_right = "images/"
    images_left = "ImagesLeft/"
    showImages = False
    save_path = "stereo_calibration.npz"
    R = np.eye(3)
    T = np.array([0.133,0,0])

    worldPoints, imgPoints, = getCorners(showImages=showImages, path=images_right)
    camera_matrix_right, dist_right, rvecs_right, tvecs_right = getParams(worldPoints, imgPoints, (1280,720))
    #removeDistortion(cameraMatrix,dist)
    right_error = checkError(worldPoints, imgPoints, rvecs_right, tvecs_right, camera_matrix_right, dist_right)

    worldPoints, imgPoints, = getCorners(showImages=showImages, path=images_left)
    camera_matrix_left, dist_left, rvecs_left, tvecs_left = getParams(worldPoints, imgPoints, (1280,720))
    #removeDistortion(cameraMatrix,dist,path="ImagesLeft/")
    left_error = checkError(worldPoints, imgPoints, rvecs_left, tvecs_left, camera_matrix_left, dist_left)

    np.savez(
        save_path,
        camera_matrix_right = camera_matrix_right,
        camera_matrix_left = camera_matrix_left,
        dist_right = dist_right,
        dist_left = dist_left,
        R = R,
        T = T
    )




