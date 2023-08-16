import face_recognition
import cv2


#TODO 
# currently frame seems to be blue
# add step so can read not every frame (increase speed)
# add minimum face size such that faces of a small size can be optionally ignored.
# convert frame to greyscale to improve accuracy (convert back when saving image)

#return list of numpy arrays
def get_all_faces(video_filepath:str,time_step:int = 1) -> list[any]: 
    """"""
    cap = cv2.VideoCapture(video_filepath)

    # if failed to load video
    if not cap.isOpened():
        raise Exception(f"failed to open video {video_filepath}")
    
    
    #calculate frame step
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(time_step * cap_fps)


    distinctive_face_encodings = []
    distinctive_face_images = []

    # itterate through each frame of video and check for new faces
    while cap.isOpened():
        success,frame = cap.read()

        if not success:
            break

        if not cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_step == 0:
            continue

        # fetch face-recognition information about the current frame
        frame_face_locations = face_recognition.face_locations(img=frame)
        frame_face_encodings = face_recognition.face_encodings(face_image=frame,known_face_locations=frame_face_locations)

        for i,face_encoding in enumerate(frame_face_encodings):

            # compare face-encodings within current frame, against already identified encodings
            matches = face_recognition.compare_faces(distinctive_face_encodings,face_encoding)
            
            if not True in matches:
                # face encoding is new and distinctive from previous
                distinctive_face_encodings.append(face_encoding)

                # crop frame (colored) to only face
                y1,x2,y2,x1 = frame_face_locations[i]
                frame_cropped_to_face = frame[y1:y2,x1:x2]
                distinctive_face_images.append(frame_cropped_to_face)

    return (distinctive_face_encodings,distinctive_face_images)


if __name__ == "__main__":
    get_all_faces("media\\clip.mp4")