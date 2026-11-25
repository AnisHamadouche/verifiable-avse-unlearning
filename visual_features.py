"""P4 optional MediaPipe face-mesh mouth crop, OpenCV fallback handled by caller."""
import numpy as np

# MediaPipe classic face mesh indices for lip boundary (optional extra dependency).
LIP_IDS=(61,291,0,17,13,14,78,308,81,311,178,402)

def lip_landmark_crop(bgr, size=48):
    import cv2
    try:
        import mediapipe as mp
        if not hasattr(mp,'solutions'):return None
        mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True)
        try:result=mesh.process(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        finally:mesh.close()
        if not result.multi_face_landmarks:return None
        pts=result.multi_face_landmarks[0].landmark
        h,w=bgr.shape[:2];xs=[pts[i].x*w for i in LIP_IDS];ys=[pts[i].y*h for i in LIP_IDS]
        cx=(min(xs)+max(xs))/2;cy=(min(ys)+max(ys))/2
        half=max(max(xs)-min(xs),max(ys)-min(ys))*0.85
        x0=max(0,int(cx-half));x1=min(w,int(cx+half))
        y0=max(0,int(cy-half));y1=min(h,int(cy+half))
        if x1<=x0 or y1<=y0:return None
        roi=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        return cv2.resize(roi,(size,size),interpolation=cv2.INTER_AREA)
    except (ImportError,AttributeError):return None
