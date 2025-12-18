import cv2
import numpy as np
import math

class VirtualWhiteboardEnhanced:
    def __init__(self):
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Crear lienzo virtual
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.canvas[:] = (255, 255, 255)  # Fondo blanco
        
        # Variables para seguimiento
        self.previous_x, self.previous_y = -1, -1
        
        # Modos de dibujo
        self.modes = ["free_draw", "circle", "rectangle", "triangle", "line"]
        self.current_mode = 0
        
        # Color activo para seguimiento
        self.active_color = {
            "name": "Rojo",
            "lower": np.array([0, 100, 100]),
            "upper": np.array([10, 255, 255]),
            "draw_color": (0, 0, 255)
        }
        
        # Configuración de colores
        self.colors = {
            "Rojo": {
                "lower": np.array([0, 100, 100]),
                "upper": np.array([10, 255, 255]),
                "draw_color": (0, 0, 255)
            },
            "Verde": {
                "lower": np.array([40, 100, 100]),
                "upper": np.array([80, 255, 255]),
                "draw_color": (0, 255, 0)
            },
            "Azul": {
                "lower": np.array([100, 100, 100]),
                "upper": np.array([130, 255, 255]),
                "draw_color": (255, 0, 0)
            }
        }
        
        # Estado del programa
        self.drawing = False
        self.show_mask = False
        self.show_info = True
        self.brush_size = 5
        self.figure_size = 50
        
        # Para control de movimiento y rotación
        self.movement_vector = [0, 0]
        self.last_landmark = None
        self.figure_rotation = 0
        
        # Nuevas variables para control de rotación mejorado
        self.rotation_mode = "manual"  # "manual", "auto", "gesture"
        self.rotation_speed = 2.0
        self.gesture_rotation_center = None
        
        print("=== PIZARRA VIRTUAL MEJORADA ===")
        print("Controles básicos:")
        print("  'm' - Cambiar modo de dibujo")
        print("  'c' - Cambiar color de seguimiento")
        print("  'espacio' - Limpiar pizarra")
        print("  'q' - Salir")
        print("\nControles de ROTACION:")
        print("  'r' - Rotar 15° (modo manual)")
        print("  'R' - Cambiar modo de rotación (manual/auto/gesture)")
        print("  '+' - Aumentar velocidad de rotación automática")
        print("  '-' - Disminuir velocidad de rotación automática")
        print("  'g' - Establecer centro para gesto circular")

    def get_landmark(self, frame):
        """Detecta el objeto del color activo y devuelve su centroide"""
        # Convertir a espacio HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para el color activo
        mask = cv2.inRange(hsv, self.active_color["lower"], self.active_color["upper"])
        
        # Operaciones morfológicas para reducir ruido
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calcular el centroide y área
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(largest_contour)
                
                # Calcular vector de movimiento
                if self.last_landmark is not None:
                    self.movement_vector = [cx - self.last_landmark[0], cy - self.last_landmark[1]]
                self.last_landmark = (cx, cy)
                
                return (cx, cy), mask, largest_contour, area
        
        return None, mask, None, 0

    def update_rotation(self, x, y, area):
        """Actualiza la rotación según el modo seleccionado"""
        if self.modes[self.current_mode] not in ["rectangle", "triangle", "line"]:
            return
            
        if self.rotation_mode == "manual":
            # La rotación se controla solo con la tecla 'r'
            pass
            
        elif self.rotation_mode == "auto":
            # Rotación automática continua
            self.figure_rotation = (self.figure_rotation + self.rotation_speed) % 360
            
        elif self.rotation_mode == "gesture":
            # Rotación basada en movimiento circular
            if self.gesture_rotation_center is not None:
                # Calcular ángulo respecto al centro del gesto
                dx = x - self.gesture_rotation_center[0]
                dy = y - self.gesture_rotation_center[1]
                if dx != 0 or dy != 0:
                    self.figure_rotation = math.degrees(math.atan2(dy, dx))
        
        # Control por área (opcional): objetos más grandes rotan más
        if area > 3000 and self.rotation_mode == "auto":
            rotation_factor = min(area / 5000, 3.0)
            self.figure_rotation = (self.figure_rotation + rotation_factor) % 360

    def draw_rotated_rectangle(self, x, y, color):
        """Dibuja un rectángulo rotado"""
        width = self.figure_size * 2
        height = self.figure_size
        
        # Crear matriz de rotación
        angle_rad = math.radians(self.figure_rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Puntos del rectángulo no rotado (centrado en origen)
        half_w = width / 2
        half_h = height / 2
        points = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        
        # Aplicar rotación
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_points = np.dot(points, rotation_matrix.T)
        
        # Trasladar a posición (x, y)
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y
        
        # Convertir a enteros
        rotated_points = rotated_points.astype(int)
        
        # Dibujar
        cv2.fillPoly(self.canvas, [rotated_points], color)
        cv2.polylines(self.canvas, [rotated_points], True, (0, 0, 0), 2)

    def draw_on_canvas(self, x, y, area):
        """Dibuja en el lienzo según el modo actual"""
        mode = self.modes[self.current_mode]
        color = self.active_color["draw_color"]
        
        # Actualizar rotación
        self.update_rotation(x, y, area)
        
        if mode == "free_draw":
            # Modo trazado libre
            if self.previous_x != -1 and self.previous_y != -1:
                cv2.line(self.canvas, (self.previous_x, self.previous_y), (x, y), 
                        color, self.brush_size)
            self.previous_x, self.previous_y = x, y
            
        elif mode == "circle":
            # Dibujar círculo
            cv2.circle(self.canvas, (x, y), self.figure_size, color, -1)
            cv2.circle(self.canvas, (x, y), self.figure_size, (0, 0, 0), 2)
            
        elif mode == "rectangle":
            # Dibujar rectángulo rotado
            self.draw_rotated_rectangle(x, y, color)
            
        elif mode == "triangle":
            # Dibujar triángulo con posible rotación
            height = int(self.figure_size * math.sqrt(3) / 2)
            
            # Crear triángulo no rotado
            triangle_pts = np.array([
                [x, y - height // 2],
                [x - self.figure_size // 2, y + height // 2],
                [x + self.figure_size // 2, y + height // 2]
            ], np.int32)
            
            # Aplicar rotación si es necesario
            if abs(self.figure_rotation) > 0.1:
                center = np.array([x, y])
                angle_rad = math.radians(self.figure_rotation)
                
                # Crear matriz de rotación
                rotation_matrix = np.array([
                    [math.cos(angle_rad), -math.sin(angle_rad)],
                    [math.sin(angle_rad), math.cos(angle_rad)]
                ])
                
                # Aplicar rotación a cada punto
                for i in range(3):
                    point = triangle_pts[i] - center
                    rotated = np.dot(rotation_matrix, point)
                    triangle_pts[i] = rotated + center
                triangle_pts = triangle_pts.astype(int)
            
            cv2.fillPoly(self.canvas, [triangle_pts], color)
            cv2.polylines(self.canvas, [triangle_pts], True, (0, 0, 0), 2)
            
        elif mode == "line":
            # Dibujar línea con control de ángulo
            length = self.figure_size * 2
            
            # Determinar ángulo según el modo de rotación
            if self.rotation_mode == "gesture" and self.gesture_rotation_center is not None:
                # Ángulo desde el centro del gesto
                dx = x - self.gesture_rotation_center[0]
                dy = y - self.gesture_rotation_center[1]
                if dx != 0 or dy != 0:
                    angle = math.degrees(math.atan2(dy, dx))
                else:
                    angle = self.figure_rotation
            elif self.movement_vector[0] != 0 or self.movement_vector[1] != 0:
                # Ángulo basado en movimiento reciente
                angle = math.degrees(math.atan2(self.movement_vector[1], self.movement_vector[0]))
            else:
                # Usar ángulo almacenado
                angle = self.figure_rotation
            
            # Calcular punto final
            end_x = int(x + length * math.cos(math.radians(angle)))
            end_y = int(y + length * math.sin(math.radians(angle)))
            
            # Dibujar línea
            cv2.line(self.canvas, (x, y), (end_x, end_y), color, self.brush_size * 2)
            
            # Marcar extremo
            cv2.circle(self.canvas, (end_x, end_y), 5, (0, 0, 0), -1)
        
        # Si no estamos en modo de dibujo libre, reiniciar coordenadas previas
        if mode != "free_draw":
            self.previous_x, self.previous_y = -1, -1

    def update_figure_size(self):
        """Actualiza el tamaño de la figura basado en movimiento"""
        if self.last_landmark and self.modes[self.current_mode] != "free_draw":
            speed = math.sqrt(self.movement_vector[0]**2 + self.movement_vector[1]**2)
            if speed > 5:
                new_size = min(150, max(20, int(speed * 1.5)))
                self.figure_size = int(self.figure_size * 0.7 + new_size * 0.3)

    def run(self):
        """Bucle principal del programa"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: No se puede capturar video. Verifica tu cámara.")
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            
            # Obtener landmark (centroide del objeto detectado)
            landmark, mask, contour, area = self.get_landmark(frame)
            
            # Actualizar tamaño basado en movimiento
            self.update_figure_size()
            
            # Si se detecta un landmark
            if landmark:
                x, y = landmark
                
                # Dibujar en el lienzo
                self.draw_on_canvas(x, y, area)
                
                # Dibujar información en el frame de video
                if self.show_info:
                    # Dibujar círculo en el landmark
                    cv2.circle(frame_copy, (x, y), 10, self.active_color["draw_color"], -1)
                    cv2.circle(frame_copy, (x, y), 10, (255, 255, 255), 2)
                    
                    # Dibujar contorno del objeto
                    if contour is not None:
                        cv2.drawContours(frame_copy, [contour], -1, (0, 255, 255), 2)
                    
                    # Mostrar información de rotación
                    cv2.putText(frame_copy, f"Rotacion: {self.figure_rotation:.1f}°", 
                               (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Indicador visual de dirección/rotación
                    radius = 25
                    end_x = int(x + radius * math.cos(math.radians(self.figure_rotation)))
                    end_y = int(y + radius * math.sin(math.radians(self.figure_rotation)))
                    cv2.arrowedLine(frame_copy, (x, y), (end_x, end_y), 
                                   (0, 255, 255), 2, tipLength=0.3)
                    
                    # Mostrar centro de gesto si está activo
                    if self.gesture_rotation_center is not None:
                        cx, cy = self.gesture_rotation_center
                        cv2.circle(frame_copy, (cx, cy), 8, (255, 0, 255), -1)
                        cv2.line(frame_copy, (cx, cy), (x, y), (255, 0, 255), 1)
            
            # Mostrar máscara si está activado
            if self.show_mask:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_colored, "Mascara HSV", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Mascara", mask_colored)
            
            # Combinar el lienzo con el frame de video
            combined = cv2.addWeighted(frame_copy, 0.7, self.canvas, 0.3, 0)
            
            # Añadir información de estado
            if self.show_info:
                mode_text = f"Modo: {self.modes[self.current_mode].replace('_', ' ').title()}"
                color_text = f"Color: {self.active_color['name']}"
                size_text = f"Tamaño: {self.brush_size if self.modes[self.current_mode]=='free_draw' else self.figure_size}"
                rot_mode_text = f"Rot Mode: {self.rotation_mode}"
                
                cv2.putText(combined, mode_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, color_text, (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(self.active_color["draw_color"]), 2)
                cv2.putText(combined, size_text, (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, rot_mode_text, (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
                
                # Mostrar controles básicos
                cv2.putText(combined, "Controles: m-Modo, c-Color, esp-Limpiar, q-Salir", 
                           (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(combined, "Rotacion: r(+15°), R(Cambiar modo), +/-(Velocidad), g(Centro gesto)", 
                           (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mostrar ventana principal
            cv2.imshow("Pizarra Virtual - Seguimiento por Color", combined)
            
            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Cambiar modo de dibujo
                self.current_mode = (self.current_mode + 1) % len(self.modes)
                print(f"Modo cambiado a: {self.modes[self.current_mode]}")
            elif key == ord('c'):
                # Cambiar color de seguimiento
                color_names = list(self.colors.keys())
                current_idx = color_names.index(self.active_color["name"]) if self.active_color["name"] in color_names else 0
                next_idx = (current_idx + 1) % len(color_names)
                self.active_color = {
                    "name": color_names[next_idx],
                    "lower": self.colors[color_names[next_idx]]["lower"],
                    "upper": self.colors[color_names[next_idx]]["upper"],
                    "draw_color": self.colors[color_names[next_idx]]["draw_color"]
                }
                print(f"Color cambiado a: {self.active_color['name']}")
            elif key == ord('b'):
                # Cambiar tamaño
                if self.modes[self.current_mode] == "free_draw":
                    self.brush_size = min(20, self.brush_size + 1) if self.brush_size < 5 else 3
                    print(f"Tamaño del pincel: {self.brush_size}")
                else:
                    self.figure_size = min(150, self.figure_size + 10) if self.figure_size < 100 else 20
                    print(f"Tamaño de figura: {self.figure_size}")
            elif key == ord('s'):
                # Mostrar/ocultar máscara
                self.show_mask = not self.show_mask
                if not self.show_mask:
                    cv2.destroyWindow("Mascara")
            elif key == ord('i'):
                # Mostrar/ocultar información
                self.show_info = not self.show_info
            elif key == ord('r'):
                # Rotación manual de 15°
                self.figure_rotation = (self.figure_rotation + 15) % 360
                print(f"Rotación manual: {self.figure_rotation}°")
            elif key == ord('R'):  # Mayúscula
                # Cambiar modo de rotación
                modes = ["manual", "auto", "gesture"]
                current_idx = modes.index(self.rotation_mode)
                self.rotation_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Modo rotación cambiado a: {self.rotation_mode}")
            elif key == ord('+') or key == ord('='):
                # Aumentar velocidad de rotación
                self.rotation_speed = min(10.0, self.rotation_speed + 0.5)
                print(f"Velocidad rotación: {self.rotation_speed}")
            elif key == ord('-') or key == ord('_'):
                # Disminuir velocidad de rotación
                self.rotation_speed = max(0.1, self.rotation_speed - 0.5)
                print(f"Velocidad rotación: {self.rotation_speed}")
            elif key == ord('g'):
                # Establecer centro para gesto de rotación
                if self.last_landmark:
                    self.gesture_rotation_center = self.last_landmark
                    print(f"Centro de gesto establecido en {self.gesture_rotation_center}")
            elif key == 32:  # Tecla espacio
                # Limpiar pizarra
                self.canvas[:] = (255, 255, 255)
                self.gesture_rotation_center = None
                print("Pizarra limpiada")
        
        # Liberar recursos
        self.cap.release()
        cv2.destroyAllWindows()

# Función principal simplificada
if __name__ == "__main__":
    # Crear y ejecutar la pizarra virtual
    whiteboard = VirtualWhiteboardEnhanced()
    whiteboard.run()
