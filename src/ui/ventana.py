import cv2
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QFileDialog, QMessageBox, QMenu, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtCore import Qt

# Importamos tus módulos de lógica
from src.logic import filtros
from src.logic.gestor_estado import GestorEstado

from src.logic import analisis 
from src.ui.ventanas_aux import VentanaHistograma, VentanaCanales
from src.logic import colores
from src.logic import mapas


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analysis App")
        self.resize(900, 700)

        # Rutas y Gestor
        base_dir = os.getcwd()
        self.ruta_data = os.path.join(base_dir, "data")
        self.ruta_salidas = os.path.join(base_dir, "salidas")
        os.makedirs(self.ruta_salidas, exist_ok=True)
        
        self.gestor = GestorEstado()
        
        # Variables de estado
        self.imagen_original = None
        self.imagen_mostrada = None

        # 1. Crear Menús (Barra superior)
        self.crear_menus()

        # 2. Interfaz Central (Visor)
        self.init_ui()

    def init_ui(self):
        """Configura los visores (Original vs Resultado)"""
        # Usamos un layout horizontal para poner las imágenes lado a lado
        self.layout_visores = QHBoxLayout()
        
        # --- Visor Izquierdo (Original / Anterior) ---
        self.visor_izq = QLabel("Original")
        self.visor_izq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visor_izq.setStyleSheet("border: 1px dashed #555;")
        self.visor_izq.setScaledContents(False) # Importante para evitar efecto lupa
        
        # --- Visor Derecho (Resultado) ---
        self.visor_der = QLabel("Resultado")
        self.visor_der.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visor_der.setStyleSheet("border: 2px solid #0078d7;") # Borde azul para destacar
        self.visor_der.setScaledContents(False)

        # Política de tamaño para evitar que exploten
        from PyQt6.QtWidgets import QSizePolicy
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.visor_izq.setSizePolicy(policy)
        self.visor_der.setSizePolicy(policy)

        # Agregamos al layout
        self.layout_visores.addWidget(self.visor_izq)
        self.layout_visores.addWidget(self.visor_der)
        
        # Por defecto, ocultamos el izquierdo hasta que haya un cambio
        self.visor_izq.hide()

        # Widget contenedor
        widget_central = QWidget()
        widget_central.setLayout(self.layout_visores)
        self.setCentralWidget(widget_central)

    def crear_menus(self):
        barra_menu = self.menuBar()
        
        # 1. ARCHIVO
        menu_archivo = barra_menu.addMenu("Archivo")
        menu_archivo.addAction("Cargar Imagen", self.cargar_imagen)
        menu_archivo.addAction("Guardar Imagen (PNG)", self.guardar_imagen)
        
        # 2. EDICIÓN (Undo/Redo)
        menu_edicion = barra_menu.addMenu("Edición")
        menu_edicion.addAction("Atrás", self.accion_atras)
        menu_edicion.addAction("Adelante", self.accion_adelante)
        menu_edicion.addSeparator()
        menu_edicion.addAction("Restablecer", self.accion_restablecer)

        # 3. VER (Histograma y Canales)
        menu_ver = barra_menu.addMenu("Ver")
        menu_ver.addAction("Mostrar Histograma", self.mostrar_histograma) 
        menu_ver.addAction("Mostrar Canales", self.mostrar_canales)
        menu_ver.addAction("Componentes Conexas", self.mostrar_componentes)

        # 4. MODELOS DE COLOR
        menu_color = barra_menu.addMenu("Modelos Color")
        menu_color.addAction("RGB", lambda: self.aplicar_modelo("RGB"))
        menu_color.addAction("Escala de Grises", lambda: self.aplicar_modelo("GRAY"))
        menu_color.addAction("Binarizar", lambda: self.aplicar_modelo("BINARY"))
        menu_color.addAction("HSV", lambda: self.aplicar_modelo("HSV"))
        menu_color.addAction("CMYK", lambda: self.aplicar_modelo("CMYK"))

        # 5. ARITMÉTICA
        menu_aritmetica = barra_menu.addMenu("Aritmética")
        menu_aritmetica.addAction("Sumar", lambda: self.preparar_operacion_doble("SUMA"))
        menu_aritmetica.addAction("Restar", lambda: self.preparar_operacion_doble("RESTA"))
        menu_aritmetica.addAction("Multiplicar", lambda: self.preparar_operacion_doble("MULT"))
        menu_aritmetica.addAction("Invertir", lambda: self.aplicar_operacion_simple("INVERTIR"))

        # 6. LÓGICA
        menu_logica = barra_menu.addMenu("Lógica")
        menu_logica.addAction("OR", lambda: self.preparar_operacion_doble("OR"))
        menu_logica.addAction("AND", lambda: self.preparar_operacion_doble("AND"))
        menu_logica.addAction("NOT", lambda: self.aplicar_operacion_simple("NOT"))

        # 7. MAPAS DE COLOR
        menu_mapas = barra_menu.addMenu("Mapas Color")
        menu_mapas.addAction("JET", lambda: self.aplicar_mapa_color("JET"))
        menu_mapas.addAction("HOT", lambda: self.aplicar_mapa_color("HOT"))
        menu_mapas.addAction("OCEAN", lambda: self.aplicar_mapa_color("OCEAN"))
        menu_mapas.addAction("BONE", lambda: self.aplicar_mapa_color("BONE"))
        menu_mapas.addAction("PINK", lambda: self.aplicar_mapa_color("PINK"))
        menu_mapas.addAction("PROPIO 1", lambda: self.aplicar_mapa_color("PROPIO 1"))
        menu_mapas.addAction("PROPIO 2", lambda: self.aplicar_mapa_color("PROPIO 2"))

        # 8. MORFOLOGÍAS
        menu_morfo = barra_menu.addMenu("Morfologías")
        menu_morfo.addAction("Erosión", lambda: self.aplicar_morfologia("Erosión"))
        menu_morfo.addAction("Dilatación", lambda: self.aplicar_morfologia("Dilatación"))
        menu_morfo.addAction("Apertura", lambda: self.aplicar_morfologia("Apertura"))
        menu_morfo.addAction("Apertura EX", lambda: self.aplicar_morfologia("Apertura EX"))
        menu_morfo.addAction("Cierre", lambda: self.aplicar_morfologia("Cierre"))
        menu_morfo.addAction("Cierre EX", lambda: self.aplicar_morfologia("Cierre EX"))

        # 9. FILTRADO
        menu_filtros = barra_menu.addMenu("Filtrado")
        menu_filtros.addAction("Pasa Bajas", lambda: self.aplicar_filtro_frecuencia("Pasa Bajas"))
        menu_filtros.addAction("Pasa Altas", lambda: self.aplicar_filtro_frecuencia("Pasa Altas"))
        menu_filtros.addAction("Compresión DTC Baja", lambda: self.aplicar_filtro_frecuencia("Compresión DTC Baja"))
        menu_filtros.addAction("Compresión DTC Alta", lambda: self.aplicar_filtro_frecuencia("Compresión DTC Alta"))

    # ==========================================
    #              FUNCIONES LÓGICAS 
    # ==========================================

    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", self.ruta_data, "Imagenes (*.png *.jpg *.bmp *.tif)")
        if archivo:
            img = cv2.imread(archivo)
            if img is None:
                QMessageBox.critical(self, "Error", "No se pudo leer la imagen.")
                return
            
            # Reset total
            self.gestor.reiniciar()
            self.modelo_actual = "RGB"  # Reset del modelo de color
            
            # Guardamos la original limpia
            self.imagen_original = img
            # La mostrada inicial es una copia de la original
            self.imagen_mostrada = img.copy()
            
            # Actualizamos interfaz
            self.actualizar_visores()

    def guardar_imagen(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "No hay imagen para guardar.")
            return

        nombre_archivo = f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ruta_completa = os.path.join(self.ruta_salidas, nombre_archivo)
        
        cv2.imwrite(ruta_completa, self.imagen_mostrada)
        QMessageBox.information(self, "Guardado", f"Imagen guardada en:\n{ruta_completa}")

    def accion_atras(self):
        imagen_anterior = self.gestor.deshacer(self.imagen_mostrada)
        
        if imagen_anterior is not None:
            self.imagen_mostrada = imagen_anterior
            # Reseteamos el modelo a RGB por seguridad para evitar conflictos con histogramas
            self.modelo_actual = "RGB"
            # Actualizamos la interfaz de doble visor
            self.actualizar_visores() 

    def accion_adelante(self):
        imagen_siguiente = self.gestor.rehacer(self.imagen_mostrada)
        
        if imagen_siguiente is not None:
            self.imagen_mostrada = imagen_siguiente
            self.modelo_actual = "RGB"
            self.actualizar_visores()

    def accion_restablecer(self):
        if self.imagen_original is not None:
            self.gestor.reiniciar()
            self.imagen_mostrada = self.imagen_original.copy()
            self.modelo_actual = "RGB"
            self.actualizar_visores()


    def actualizar_visores(self):
        """Muestra las imágenes en los visores correspondientes"""
        if self.imagen_mostrada is None:
            return

        # Lógica: 
        # Si NO hay historial (es la primera carga), solo mostramos Resultado (derecha) y ocultamos Original (izq).
        # Si SÍ hay historial (ya editamos algo), mostramos ambas.
        
        if not self.gestor.historial:
            # Caso inicial: Solo una imagen
            self.visor_izq.hide()
            self.mostrar_en_label(self.visor_der, self.imagen_mostrada)
        else:
            # Caso edición: Mostramos comparación
            self.visor_izq.show()
            self.mostrar_en_label(self.visor_izq, self.imagen_original) # Siempre mostramos la original base a la izquierda
            self.mostrar_en_label(self.visor_der, self.imagen_mostrada)

    def mostrar_en_label(self, label, img_cv):
        """Convierte OpenCV -> Qt y pone la imagen en el label"""
        if img_cv is None: return
        
        # Detección de formato para conversión
        if len(img_cv.shape) == 2: # Grises/Binario
            h, w = img_cv.shape
            fmt = QImage.Format.Format_Grayscale8
            bytes_line = w
            q_img = QImage(img_cv.data, w, h, bytes_line, fmt)
            
        else: # Color
            h, w, c = img_cv.shape
            # Convertimos BGR a RGB para visualizar correctamente
            # PERO si es una simulación CMYK o HSV rara, igual la mostramos como RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
            bytes_line = 3 * w
            q_img = QImage(img_rgb.data, w, h, bytes_line, fmt)

        pixmap = QPixmap.fromImage(q_img)
        
        # Escalado suave
        label.setPixmap(pixmap.scaled(
            label.width(), label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # ==========================================
    #           FUNCIONES DE ANÁLISIS
    # ==========================================

    def mostrar_histograma(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return
            
        # Nos aseguramos de tener un modelo definido, si no, RGB por defecto
        if not hasattr(self, 'modelo_actual'):
            self.modelo_actual = "RGB"

        # 1. Calculamos la gráfica pasando el MODELO ACTUAL
        fig = analisis.calcular_histograma(self.imagen_mostrada, self.modelo_actual)
        
        # 2. Abrimos la subventana
        dialogo = VentanaHistograma(fig)
        dialogo.exec() 

    def mostrar_canales(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        # Validación por seguridad
        if not hasattr(self, 'modelo_actual'):
            self.modelo_actual = "RGB"
            
        # Pasamos el modelo para que sepa cómo interpretar los datos
        canales = analisis.separar_canales(self.imagen_mostrada, self.modelo_actual)
        
        dialogo = VentanaCanales(canales)
        dialogo.exec()


    def mostrar_componentes(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        # Llamamos a la lógica
        img_etiquetada, cantidad = analisis.etiquetar_componentes(self.imagen_mostrada)
        
        QMessageBox.information(self, "Análisis", f"Se encontraron {cantidad} objetos (componentes conexas).")
        
        # Opción B: Abrir en ventana aparte
        # Nota: Asegúrate de tener importada VentanaCanales al inicio del archivo
        # from src.ui.ventanas_aux import VentanaCanales 
        dialogo = VentanaCanales([("Mapa de Etiquetas", img_etiquetada)])
        dialogo.setWindowTitle(f"Componentes Conexas ({cantidad} objetos)")
        dialogo.exec()

    # ==========================================
    #           TRANSFORMACIONES DE COLOR
    # ==========================================

    def aplicar_modelo(self, modelo):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        self.gestor.guardar_estado(self.imagen_mostrada)
        
        # Guardamos el modelo para que el histograma sepa qué mostrar
        self.modelo_actual = modelo 
        
        resultado = colores.aplicar_modelo(self.imagen_mostrada, modelo)
        self.imagen_mostrada = resultado
        
        # Actualizamos ambos visores (Original vs Resultado)
        self.actualizar_visores()

    # Pegar esto dentro de class VentanaPrincipal:

    def aplicar_mapa_color(self, mapa):
        # 1. Validación de seguridad
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        self.gestor.guardar_estado(self.imagen_mostrada)
        self.modelo_actual = "RGB" 
        resultado = mapas.aplicar_mapa_color(self.imagen_mostrada, mapa)
        self.imagen_mostrada = resultado
        self.actualizar_visores()

    # ==========================================
    #       PLACEHOLDERS (pendientes)
    # ==========================================
    
    def aplicar_operacion_simple(self, operacion):
        print(f"Operación simple: {operacion}")

    def preparar_operacion_doble(self, operacion):
        print(f"Preparando operación doble: {operacion}")

    def aplicar_morfologia(self, operacion):
        print(f"Morfología: {operacion}")

    def aplicar_filtro_frecuencia(self, filtro):
        print(f"Filtro frecuencia: {filtro}")

