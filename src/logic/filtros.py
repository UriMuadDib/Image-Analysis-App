import cv2

def convertir_gris(imagen):
    """
    Recibe una imagen (matriz numpy).
    Devuelve la imagen en escala de grises.
    """
    if imagen is None:
        return None
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def detectar_bordes(imagen):
    """
    Aplica Canny para detectar bordes.
    """
    if imagen is None:
        return None
    # Convertimos a gris primero por buena práctica
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gris, 100, 200)
