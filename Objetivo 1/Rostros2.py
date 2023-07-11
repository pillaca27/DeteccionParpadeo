import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils

# Función para calcular el aspect ratio del ojo
def eye_aspect_ratio(eye):
    # Distancia euclidiana entre los puntos verticales del ojo (vertical eye landmarks)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Distancia euclidiana entre los puntos horizontal del ojo (horizontal eye landmarks)
    C = distance.euclidean(eye[0], eye[3])
    
    # Calculamos el aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Parámetros
umbral_parapdeo = 0.2
contador_parapdeos = 0
consec_parapdeos = 3
nivel_confiabilidad = 0

# Importamos el modelo facial de Dlib para detección de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Iniciamos la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Leemos los frames
    ret, frame = cap.read()

    # Si hay error
    if not ret:
        break

    # Realizamos conversión de formato
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    faces = detector(gray)

    for face in faces:
        # Detección de landmarks faciales
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Extracción de coordenadas de los ojos
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Cálculo del aspect ratio de los ojos
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Promedio de ambos ojos
        avg_ear = (left_ear + right_ear) / 2.0

        # Dibujar contorno de ojos
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # Detección de parpadeo
        if avg_ear < umbral_parapdeo:
            contador_parapdeos += 1
        else:
            if contador_parapdeos >= consec_parapdeos:
                nivel_confiabilidad += 1
                print("Parpadeo detectado - Nivel de confiabilidad:", nivel_confiabilidad)
            contador_parapdeos = 0

    cv2.imshow("DETECCION DE PARPADEOS", frame)

    # Salir con la tecla 'Esc'
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
