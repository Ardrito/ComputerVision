import cv2 as cv
import numpy as np
import glob

def getCorners(chessboardSize: tuple[int,int] = (10,7), frameSize: tuple[int,int]=(1280,720), 
               path: str = "images/", fileType: str = ".jpeg", showImages: bool = False):
    '''
    Find corners of squares in images of checkerboards from a folder of images
    
    Recommended between 5-10 images and non square checkerboard of at least 5x4, recommended 

    path as "path/to/file/"

    Returns array (image, points, location(x,y))
    '''
    assert (chessboardSize[0] > 4) & (chessboardSize[1] > 3) & (chessboardSize[0] > chessboardSize[1]), "Chessboard should be at least 5x4 and for an M x N, M>N"
    assert (frameSize[0] > 0) & (frameSize[1] > 0), "frame size should be non zero"

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    worldP = np.zeros((chessboardSize[0] *chessboardSize[1], 3), np.float32)
    worldP[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

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




def getParams(worldPoints, imgPoints, frameSize, savePath="calibration.npz"):
    '''
    Get calibration parameters for a camera

    Saves a numpy file and returns values
    '''
    repError, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(worldPoints, imgPoints, frameSize,None,None)

    #print ("repError: " , repError)

    np.savez(
        savePath,
        repError=repError,
        cameraMatrix=cameraMatrix,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs
    )

    return cameraMatrix,dist, rvecs, tvecs

def removeDistortion(cameraMatrix, dist, path: str = "images/", fileType: str = ".jpeg"):

    '''
    Removes distortion from an image and saves it in path
    '''
    img = cv.imread(glob.glob(path+"*"+fileType)[0])
    h,w = img.shape[:2]
    newCamMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix,dist,(h,w),1,(h,w))

    undistImg = cv.undistort(img, cameraMatrix, dist, None, newCamMatrix)
    x,y,h,w = roi

    undistImg = undistImg[y:y+h, x:x+w]

    cv.imwrite("undistorted.jpeg", undistImg)

    # cv.imshow("img", img)
    # cv.imshow("undist ", undistImg)
    # cv.waitKey(10000)

def checkError(worldPoints, imgPoints, rvecs, tvecs):
    meanError = 0

    for i in range(len(worldPoints)):
        newImgPoints, _ = cv.projectPoints(worldPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgPoints[i], newImgPoints, cv.NORM_L2)/len(newImgPoints)
        meanError += error

    error = meanError/len(worldPoints)

    print ("Error is", error)

    return error





worldPoints, imgPoints, = getCorners(showImages=False)
cameraMatrix, dist, rvecs, tvecs = getParams(worldPoints, imgPoints, (1280,720))
removeDistortion(cameraMatrix,dist)
checkError(worldPoints, imgPoints, rvecs, tvecs)

# print (cameraMatrix)
# print (dist)




