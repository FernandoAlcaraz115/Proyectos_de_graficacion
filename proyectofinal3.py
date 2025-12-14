from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric, gluCylinder, gluSphere
import glfw
import cv2
import numpy as np
import sys
import math
# Variables globales para transformaciones
rotation_angle = 0.0
translation_x = 0.0
translation_y = 0.0
scale_factor = 1.0
prev_gray = None
flow_threshold = 0.5

# Configuración global
window = None  # Ventana global

# Funciones de inicialización

def init():
    """Configuración inicial de OpenGL"""
    glClearColor(0.5, 0.8, 1.0, 1.0)  # Fondo azul cielo
    glEnable(GL_DEPTH_TEST)           # Activar prueba de profundidad

    # Configuración de la perspectiva
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.0, 0.1, 100.0)  # Campo de visión más amplio
    glMatrixMode(GL_MODELVIEW)



def draw_toroid(inner_radius, outer_radius, slices, stacks):
    for i in range(slices):
        glBegin(GL_QUAD_STRIP)
        for j in range(stacks + 1):
            for k in [i, i + 1]:
                angle = 2.0 * math.pi * k / slices
                next_angle = 2.0 * math.pi * j / stacks
                x = (outer_radius + inner_radius * math.cos(next_angle)) * math.cos(angle)
                y = (outer_radius + inner_radius * math.cos(next_angle)) * math.sin(angle)
                z = inner_radius * math.sin(next_angle)
                glVertex3f(x, y, z)
        glEnd()

def draw_cylinder():
    glPushMatrix()
    glColor3f(0.6, 0.0, 0.0)  # Color marrón claro
    glTranslatef(0.0, -1.0, 0.0)  # Ajusta la posición
    glRotatef(-90, 1, 0, 0)  # Rota para orientar el cilindro verticalmente
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.2, 0.2, 5.0, 32, 32)  # Poste de la canasta más alto
    glPopMatrix()

def draw_board():
    glPushMatrix()
    glColor3f(1.0, 1.0, 1.0)  # Blanco
    glTranslatef(0.0, 3.5, -0.5)  # Posición del tablero más alto
    glScalef(1.5, 1.0, 0.1)  # Escala para darle forma de tablero
    glBegin(GL_QUADS)
    for face in [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),  # Frontal
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)       # Posterior
    ]:
        glVertex3f(*face)
    glEnd()
    glPopMatrix()

def draw_hoop():
    glPushMatrix()
    glColor3f(1.0, 0.0, 0.0)  # Rojo
    glTranslatef(0.0, 3.0, -1.1)  # Ajusta la posición del aro más alto
    glRotatef(90, 1, 0, 0)  # Rota el aro para que quede acostado
    draw_toroid(0.05, 0.5, 32, 32)  # Radio interno, radio externo, segmentos
    glPopMatrix()

def draw_light_pole():
    glPushMatrix()
    glColor3f(0.4, 0.4, 0.4)  # Gris para el poste
    glTranslatef(3.0, 0.0, 0.0)  # Posicionar el poste a un lado
    glRotatef(-90, 1, 0, 0)  # Rota para orientar el cilindro verticalmente
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.1, 0.1, 6.0, 32, 32) 
    glPopMatrix()

    glPushMatrix()
    glColor3f(1.0, 1.0, 0.0)  # Amarillo para la luz
    glTranslatef(3.0, 6.0, 0.0)  # Posicionar la esfera encima del poste
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.3, 32, 32)  # Luz esférica
    glPopMatrix()

def draw_cloud():
    glPushMatrix()
    positions = [
        (0.0, 3.5, 0.0),  
        (-0.8, 3.0, 0.0),
        (0.8, 3.0, 0.0),
        (0.0, 3.0, -0.8),
        (0.0, 3.0, 0.8)
    ]
    
    glColor3f(1.0, 1.0, 1.0)  # Color blanco para la nube
    quadric = gluNewQuadric()

    for pos in positions:
        glPushMatrix()
        glTranslatef(*pos)
        gluSphere(quadric, 1.0, 32, 32)  
        glPopMatrix()
    
    glPopMatrix()

def draw_hollow_cylinder():
    glPushMatrix()
    glColor3f(0.6, 0.6, 0.6)  # Gris para el cilindro
    glTranslatef(3.0, .5, 0.0)  
    glRotatef(90, 1, 0, 0)  # Rota para que esté vertical

    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.15, 0.15, .5, 32, 32)  # Cilindro hueco
    glPopMatrix()

def draw_snowman():
    glPushMatrix()
    glTranslatef(6.0, 0.0, 0.0)  
    
    glRotatef(180, 0.0, 1.0, 0.0) 
    
    glColor3f(1.0, 1.0, 1.0)  # Blanco para el muñeco de nieve
    
    # Esfera inferior
    gluSphere(gluNewQuadric(), 1.5, 32, 32)  # Radio más grande
    glTranslatef(0.0, 2.0, 0.0)  
    
    # Esfera del medio
    gluSphere(gluNewQuadric(), 1.2, 32, 32)
    glTranslatef(0.0, 1.5, 0.0)  
    
    # Cabeza
    gluSphere(gluNewQuadric(), 1.0, 32, 32)

   
    glPushMatrix()
    glColor3f(0.0, 0.0, 0.0)  # Negro para los ojos
    glTranslatef(-0.4, 0.5, 0.8)  # Ojo izquierdo
    gluSphere(gluNewQuadric(), 0.1, 16, 16)
    
    glTranslatef(0.8, 0.0, 0.0)  # Ojo derecho
    gluSphere(gluNewQuadric(), 0.1, 16, 16)
    glPopMatrix()

    glPushMatrix()
    glColor3f(1.0, 0.5, 0.0)  # Naranja para la nariz
    glTranslatef(0.0, 0.5, 1.1)  # Posición de la nariz
    gluSphere(gluNewQuadric(), 0.2, 16, 16)  # Nariz
    glPopMatrix()
    
    glPopMatrix()
    
#Arboles
def draw_trunk1():
    
    glPushMatrix()
    glColor3f(0.5, 0.25, 0.1)  
    glTranslatef(0.0, 0.0, 0.0)  
    glRotatef(-90, 1, 0, 0)  
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.5, 0.3, 3.0, 32, 32)  
    glPopMatrix()

def draw_foliage1():
    
    positions = [
        (0.0, 3.5, 0.0),  
        (-0.8, 3.0, 0.0),
        (0.8, 3.0, 0.0),
        (0.0, 3.0, -0.8),
        (0.0, 3.0, 0.8)
    ]
    
    glColor3f(0.2, 0.9, 0.2)  
    quadric = gluNewQuadric()

    for pos in positions:
        glPushMatrix()
        glTranslatef(*pos)
        gluSphere(quadric, 1.0, 32, 32)  
        glPopMatrix()


def draw_trunk2():
    
    glPushMatrix()
    glColor3f(0.5, 0.25, 0.1) 
    glTranslatef(0.0, 0.0, 0.0)  
    glRotatef(-90, 1, 0, 0)  
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.2, 0.2, 3.0, 32, 32)  
    glPopMatrix()

def draw_foliage2():
    
    positions = [
        (0.0, 2.5, 0.0, 2.0), 
        (0.0, 3.5, 0.0, 1.5), 
        (0.0, 4.3, 0.0, 1.0)  
    ]

    glColor3f(0.1, 0.5, 0.1) 
    quadric = gluNewQuadric()

    for x, y, z, radius in positions:
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(-90, 1, 0, 0)  
        gluCylinder(quadric, radius, 0.0, 1.5, 32, 32)  
        glPopMatrix()

def draw_trunk3():
    
    glPushMatrix()
    glColor3f(0.9, 0.9, 0.8)  
    glTranslatef(0.0, 0.0, 0.0)  
    glRotatef(-90, 1, 0, 0)  
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.25, 0.2, 4.0, 32, 32)  
    glPopMatrix()

def draw_foliage3():
    
    positions = [
        (0.0, 4.0, 0.0),  
        (-0.5, 3.5, 0.5),
        (0.5, 3.5, -0.5),
        (0.3, 3.8, 0.3),
        (-0.3, 3.8, -0.3)
    ]

    glColor3f(0.5, 0.8, 0.4)  
    quadric = gluNewQuadric()

    for pos in positions:
        glPushMatrix()
        glTranslatef(*pos)
        gluSphere(quadric, 0.5, 32, 32)  
        glPopMatrix()

def draw_trunk4():
    
    glPushMatrix()
    glColor3f(0.6, 0.3, 0.1)  
    glTranslatef(0.0, 0.0, 0.0) 
    glRotatef(-90, 1, 0, 0)  
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.3, 0.3, 2.0, 32, 32)  
    glPopMatrix()

def draw_foliage4():
    
    glPushMatrix()
    glColor3f(0.1, 0.8, 0.1)  
    glTranslatef(0.0, 2.0, 0.0)  
    quadric = gluNewQuadric()
    gluSphere(quadric, 1.0, 32, 32)  
    glPopMatrix()


def draw_trunk5():
    
    glPushMatrix()
    glColor3f(0.5, 0.3, 0.1) 
    glTranslatef(0.0, 0.0, 0.0) 
    glRotatef(-90, 1, 0, 0)  
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.3, 0.3, 3.0, 32, 32)  
    glPopMatrix()

def draw_foliage5():
    
    glPushMatrix()
    glColor3f(0.1, 0.6, 0.1)  

    
    for i in range(5):
        glPushMatrix()
        glTranslatef(0.0, 2.5 + i * 0.6, 0.0)  
        glRotatef(90, 1, 0, 0)  
        gluSphere(gluNewQuadric(), 1.0 - i * 0.2, 16, 16)  
        glPopMatrix()

    glPopMatrix()

#Casas

def draw_cube1():
    """Dibuja el cubo (base de la casa)"""
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.5, 0.2)  # Marrón para todas las caras

    # Frente
    glVertex3f(-1, 0, 1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Atrás
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)

    # Izquierda
    glVertex3f(-1, 0, -1)
    glVertex3f(-1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)

    # Derecha
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)

    # Arriba
    glColor3f(0.9, 0.6, 0.3)  # Color diferente para el techo
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Abajo
    glColor3f(0.6, 0.4, 0.2)  # Suelo más oscuro
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(-1, 0, 1)
    glEnd()

def draw_roof1():
    """Dibuja el techo (pirámide)"""
    glBegin(GL_TRIANGLES)
    glColor3f(0.9, 0.1, 0.1)  # Rojo brillante

    # Frente
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(0, 2, 0)

    # Atrás
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(0, 2, 0)

    # Izquierda
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(0, 2, 0)

    # Derecha
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(0, 2, 0)
    glEnd()

def draw_cube(x, y, z, scale_x, scale_y, scale_z, color):
    glPushMatrix()
    glTranslatef(x, y, z)
    glScalef(scale_x, scale_y, scale_z)

    glBegin(GL_QUADS)
    glColor3f(*color)  # Color personalizado para todas las caras

    # Frente
    glVertex3f(-1, 0, 1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Atrás
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)

    # Izquierda
    glVertex3f(-1, 0, -1)
    glVertex3f(-1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)

    # Derecha
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)

    # Arriba
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Abajo
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(-1, 0, 1)
    glEnd()

    glPopMatrix()

def draw_roof(x, y, z, size):
    glPushMatrix()
    glTranslatef(x, y, z)

    glBegin(GL_TRIANGLES)
    glColor3f(0.9, 0.1, 0.1)  # Rojo brillante

    # Frente
    glVertex3f(-size, 0, size)
    glVertex3f(size, 0, size)
    glVertex3f(0, size, 0)

    # Atrás
    glVertex3f(-size, 0, -size)
    glVertex3f(size, 0, -size)
    glVertex3f(0, size, 0)

    # Izquierda
    glVertex3f(-size, 0, -size)
    glVertex3f(-size, 0, size)
    glVertex3f(0, size, 0)

    # Derecha
    glVertex3f(size, 0, -size)
    glVertex3f(size, 0, size)
    glVertex3f(0, size, 0)
    glEnd()

    glPopMatrix()
    
def draw_rectangular_base():
    glBegin(GL_QUADS)
    glColor3f(0.7, 0.4, 0.2)  # Marrón claro

    # Frente
    glVertex3f(-2, 0, 1)
    glVertex3f(2, 0, 1)
    glVertex3f(2, 1, 1)
    glVertex3f(-2, 1, 1)

    # Atrás
    glVertex3f(-2, 0, -1)
    glVertex3f(2, 0, -1)
    glVertex3f(2, 1, -1)
    glVertex3f(-2, 1, -1)

    # Izquierda
    glVertex3f(-2, 0, -1)
    glVertex3f(-2, 0, 1)
    glVertex3f(-2, 1, 1)
    glVertex3f(-2, 1, -1)

    # Derecha
    glVertex3f(2, 0, -1)
    glVertex3f(2, 0, 1)
    glVertex3f(2, 1, 1)
    glVertex3f(2, 1, -1)

    # Arriba
    glColor3f(0.8, 0.5, 0.3)  
    glVertex3f(-2, 1, -1)
    glVertex3f(2, 1, -1)
    glVertex3f(2, 1, 1)
    glVertex3f(-2, 1, 1)

    # Abajo
    glColor3f(0.6, 0.4, 0.2)  
    glVertex3f(-2, 0, -1)
    glVertex3f(2, 0, -1)
    glVertex3f(2, 0, 1)
    glVertex3f(-2, 0, 1)
    glEnd()

def draw_prism_roof():
    glBegin(GL_TRIANGLES)
    glColor3f(0.9, 0.2, 0.2)  # Rojo brillante

    # Frente
    glVertex3f(-2, 1, 1)
    glVertex3f(2, 1, 1)
    glVertex3f(0, 2, 0)

    # Atrás
    glVertex3f(-2, 1, -1)
    glVertex3f(2, 1, -1)
    glVertex3f(0, 2, 0)
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.8, 0.2, 0.2)  # Rojo oscuro

    # Izquierda
    glVertex3f(-2, 1, -1)
    glVertex3f(-2, 1, 1)
    glVertex3f(0, 2, 0)
    glVertex3f(0, 2, 0)

    # Derecha
    glVertex3f(2, 1, -1)
    glVertex3f(2, 1, 1)
    glVertex3f(0, 2, 0)
    glVertex3f(0, 2, 0)
    glEnd()

def draw_door():
    glBegin(GL_QUADS)
    glColor3f(0.4, 0.2, 0.1)  # Marrón oscuro
    
    # Frente
    glVertex3f(-0.5, 0, 1.01)
    glVertex3f(0.5, 0, 1.01)
    glVertex3f(0.5, 0.6, 1.01)
    glVertex3f(-0.5, 0.6, 1.01)
    glEnd()

def draw_windows():
    glBegin(GL_QUADS)
    glColor3f(0.3, 0.7, 0.9)  # Azul para el vidrio

    # Ventana izquierda
    glVertex3f(-1.5, 0.5, 1.01)
    glVertex3f(-1.0, 0.5, 1.01)
    glVertex3f(-1.0, 0.8, 1.01)
    glVertex3f(-1.5, 0.8, 1.01)

    # Ventana derecha
    glVertex3f(1.0, 0.5, 1.01)
    glVertex3f(1.5, 0.5, 1.01)
    glVertex3f(1.5, 0.8, 1.01)
    glVertex3f(1.0, 0.8, 1.01)
    glEnd()
 
def draw_base():
    """Dibuja la base de la casa"""
    glBegin(GL_QUADS)
    glColor3f(0.6, 0.6, 0.6)  # Gris claro para la base

    # Nivel inferior
    glVertex3f(-2, 0, -2)
    glVertex3f(2, 0, -2)
    glVertex3f(2, 1, -2)
    glVertex3f(-2, 1, -2)

    glVertex3f(-2, 0, 2)
    glVertex3f(2, 0, 2)
    glVertex3f(2, 1, 2)
    glVertex3f(-2, 1, 2)

    glVertex3f(-2, 0, -2)
    glVertex3f(-2, 0, 2)
    glVertex3f(-2, 1, 2)
    glVertex3f(-2, 1, -2)

    glVertex3f(2, 0, -2)
    glVertex3f(2, 0, 2)
    glVertex3f(2, 1, 2)
    glVertex3f(2, 1, -2)
    
    glVertex3f(-2, 1, -2)
    glVertex3f(2, 1, -2)
    glVertex3f(2, 1, 2)
    glVertex3f(-2, 1, 2)
    glEnd()

def draw_second_floor():
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.8, 0.8)  # Gris más claro para el segundo piso

    # Paredes del segundo piso
    glVertex3f(-1.8, 1, -1.8)
    glVertex3f(1.8, 1, -1.8)
    glVertex3f(1.8, 2, -1.8)
    glVertex3f(-1.8, 2, -1.8)

    glVertex3f(-1.8, 1, 1.8)
    glVertex3f(1.8, 1, 1.8)
    glVertex3f(1.8, 2, 1.8)
    glVertex3f(-1.8, 2, 1.8)

    glVertex3f(-1.8, 1, -1.8)
    glVertex3f(-1.8, 1, 1.8)
    glVertex3f(-1.8, 2, 1.8)
    glVertex3f(-1.8, 2, -1.8)

    glVertex3f(1.8, 1, -1.8)
    glVertex3f(1.8, 1, 1.8)
    glVertex3f(1.8, 2, 1.8)
    glVertex3f(1.8, 2, -1.8)
    glEnd()

def draw_terrace():
    glBegin(GL_QUADS)
    glColor3f(0.3, 0.3, 0.3)  # Gris oscuro para la terraza

    # Piso de la terraza
    glVertex3f(-1.9, 2, -1.9)
    glVertex3f(1.9, 2, -1.9)
    glVertex3f(1.9, 2, 1.9)
    glVertex3f(-1.9, 2, 1.9)
    glEnd()

def draw_railings():
    glBegin(GL_LINES)
    glColor3f(0.0, 0.0, 0.0)  # Negro para las barandas

    # Barandas
    for i in range(-18, 19, 3):
        glVertex3f(i / 10.0, 2, -1.9)
        glVertex3f(i / 10.0, 2.2, -1.9)

        glVertex3f(i / 10.0, 2, 1.9)
        glVertex3f(i / 10.0, 2.2, 1.9)

        glVertex3f(-1.9, 2, i / 10.0)
        glVertex3f(-1.9, 2.2, i / 10.0)

        glVertex3f(1.9, 2, i / 10.0)
        glVertex3f(1.9, 2.2, i / 10.0)

    glEnd()

def draw_large_windows():
    
    glBegin(GL_QUADS)
    glColor3f(0.2, 0.7, 0.9)  # Azul claro para las ventanas

    # Ventanas del nivel inferior
    glVertex3f(-1.8, 0.5, -2.01)
    glVertex3f(-0.2, 0.5, -2.01)
    glVertex3f(-0.2, 0.9, -2.01)
    glVertex3f(-1.8, 0.9, -2.01)

    glVertex3f(0.2, 0.5, -2.01)
    glVertex3f(1.8, 0.5, -2.01)
    glVertex3f(1.8, 0.9, -2.01)
    glVertex3f(0.2, 0.9, -2.01)

    # Ventanas del nivel superior
    glVertex3f(-1.5, 1.5, -1.81)
    glVertex3f(1.5, 1.5, -1.81)
    glVertex3f(1.5, 1.9, -1.81)
    glVertex3f(-1.5, 1.9, -1.81)
    glEnd()


def draw_ground():
    """Dibuja un plano para representar el suelo"""
    glBegin(GL_QUADS)
    glColor3f(0.3, 0.3, 0.3)  # Gris oscuro para el suelo
    glVertex3f(-20, 0, 20)
    glVertex3f(20, 0, 20)
    glVertex3f(20, 0, -20)
    glVertex3f(-20, 0, -20)
    glEnd()

# Función principal de la escena
def draw_scene():
    global rotation_angle, translation_x, translation_y, scale_factor
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(50, 25, 50, 0, 5, 0, 0, 1, 0)
    
    glTranslatef(translation_x, translation_y, 0)
    glScalef(scale_factor, scale_factor, scale_factor)
    glRotatef(rotation_angle, 0, 1, 0)

    draw_ground()  # Dibuja el suelo
    
    glTranslatef(0,2,0)
    
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(2,0,-29)
    glScaled(3.5,3.5,3.5)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    #P3
    glTranslatef(2,0,-29)
    glScaled(3.5,3.5,3.5)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    #P4
    glTranslatef(2,0,-29)
    glScaled(3.5,3.5,3.5)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    draw_cylinder()    # Dibuja el poste
    draw_board()       # Dibuja el tablero
    draw_hoop()        # Dibuja el aro
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    #p5
    glTranslatef(120,0,110)
    glScaled(4.5,4.5,4.5)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,0)
    glScaled(4,4,4)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(-5,0,4)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(0,0,-5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(2.5,0,0)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(0,0,5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-5,0,-4)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(-2,0,3)
    draw_foliage2()
    draw_trunk2()
    
    #p6
    glTranslatef(11,0,51)
    glScaled(.85,.85,.85)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,0)
    glScaled(4,4,4)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(-5,0,4)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(0,0,-5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(2.5,0,0)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(0,0,5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-5,0,-4)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(-2,0,3)
    draw_foliage2()
    draw_trunk2()
    
    #p7
    glTranslatef(20,0,65)
    glScaled(.95,.95,.95)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(5,0,0)
    draw_foliage1()
    draw_trunk1()

    
    glTranslatef(-3,0,3.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas

    
    glTranslatef(-3.5,0,-3)
    draw_hollow_cylinder()  # Dibuja bote de basura
    
    glTranslatef(-1,0,5)
    glScaled(.4,.5,.5)
    
    glTranslatef(-1,0,-9)
    glRotated(180,0,1,0)
    
    glTranslatef(-10,0,20)
    glScaled(2.5,2.5,2.5)
    glRotated(180,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(3,0,2)
    glScaled(.5,.5,.5)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-12,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(5,.5,-1)
    glScaled(.4,.4,.4)
    glRotated(180,0,1,0)
    draw_snowman()
    
    glTranslatef(-17,-1,7)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,-47)
    glScaled(2.5,2.5,2.5)
    draw_cube(0, 0, 0, 3, 2, 3, (0.8, 0.5, 0.2))  # Cubo base principal
    draw_cube(4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3))  # Cubo adicional a la derecha
    draw_cube(-4, 0, 0, 1, 2, 1, (0.7, 0.5, 0.3)) # Cubo adicional a la izquierda
    draw_roof(0, 2, 0, 3)  # Techo más grande
    
    glTranslatef(2,0,17)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(0,0,5)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,-7)
    draw_light_pole()  # Dibuja el poste de luz
    
    glScaled(3,3,3)
    glTranslatef(-3,0,-5)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glTranslatef(-2,0,2)
    glScaled(.5,.5,.5)
    draw_base()          # Dibuja la base
    draw_large_windows() # Dibuja las ventanas grandes
    draw_second_floor()  # Dibuja el segundo piso
    draw_terrace()       # Dibuja la terraza
    draw_railings()      # Dibuja las barandas
    
    glTranslatef(2.5,0,-1)
    glScaled(.6,.6,.6)
    draw_light_pole()  # Dibuja el poste de luz
    
    glTranslatef(7,0,0)
    glScaled(1.5,1.5,1.5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-7,0,5.25)
    glScaled(1.75,1.75,1.75)
    draw_ground()  # Dibuja el suelo
    draw_cube1()    # Dibuja la base de la casa
    draw_roof1()    # Dibuja el techo
    
    glScaled(.75,.75,.75)
    glTranslatef(3,0,-2)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(-3,0,5.25)
    glRotated(90,0,1,0)
    draw_ground()          # Dibuja el suelo
    draw_rectangular_base()  # Dibuja la base de la casa
    draw_door()            # Dibuja la puerta
    draw_windows()         # Dibuja las ventanas
    draw_prism_roof()      # Dibuja el techo
    
    glTranslatef(2.5,0,3)
    glScaled(.5,.5,.5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(-5,0,0)
    draw_light_pole() 
    
    glTranslatef(-3,0,0)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(-3.55,0,-6)
    glScaled(.5,.5,.5)
    draw_cube(0, 0, 0, 5, 3, 5, (0.8, 0.5, 0.2))  # Casa principal
    draw_roof(0, 3, 0, 5)  # Techo
    draw_cube(-5, 0, 0, 3, 2, 5, (0.7, 0.7, 0.7))  # Garaje
    draw_cube(1.5, 5, 0, 0.5, 1.5, 0.5, (0.6, 0.6, 0.6))  # Chimenea
    
    glTranslatef(-15,0,0)
    glScaled(4,4,4)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(-5,0,4)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(0,0,-5)
    draw_foliage3()
    draw_trunk3()
    
    glTranslatef(2.5,0,0)
    draw_foliage4()
    draw_trunk4()
    
    glTranslatef(0,0,5)
    draw_foliage5()
    draw_trunk5()
    
    glTranslatef(-5,0,-4)
    draw_foliage1()
    draw_trunk1()
    
    glTranslatef(-2,0,3)
    draw_foliage2()
    draw_trunk2()
    
    glTranslatef(0,7,0)
    glScaled(5,3,5)
    draw_cloud()
    
    glTranslatef(7,0,10)
    draw_cloud()
    
    glTranslatef(-7,0,10)
    draw_cloud()
    
    glTranslatef(7,0,-5)
    draw_cloud()
    
    glfw.swap_buffers(window)

# Procesar movimientos de cámara
def process_camera():
    global rotation_angle, translation_x, translation_y, scale_factor, prev_gray

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        print("Error: No se pudo acceder a la cámara.")
        return

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer un frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        # Dibujar los rectángulos
        cv2.rectangle(frame, (0, 0), (int(width * 0.3), int(height * 0.3)), (0, 255, 0), 2)  # Rotación
        cv2.rectangle(frame, (int(width * 0.7), 0), (width, int(height * 0.3)), (255, 0, 0), 2)  # Traslación
        cv2.rectangle(frame, (0, int(height * 0.7)), (int(width * 0.3), height), (0, 0, 255), 2)  # Escalamiento

        roi_rot = gray[:int(height * 0.3), :int(width * 0.3)]
        flow_rot = cv2.calcOpticalFlowFarneback(prev_gray[:int(height * 0.3), :int(width * 0.3)], roi_rot, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x_rot = np.mean(flow_rot[..., 0])

        roi_trans = gray[:int(height * 0.3), int(width * 0.7):]
        flow_trans = cv2.calcOpticalFlowFarneback(prev_gray[:int(height * 0.3), int(width * 0.7):], roi_trans, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x_trans = np.mean(flow_trans[..., 0])
        flow_y_trans = np.mean(flow_trans[..., 1])

        roi_scale = frame[int(height * 0.7):, :int(width * 0.3)]
        gray_scale = cv2.cvtColor(roi_scale, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_scale, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, _, w, _ = cv2.boundingRect(largest_contour)
            hand_position = x + w / 2
            rec_width = roi_scale.shape[1]
            normalized_position = hand_position / rec_width
            if normalized_position > 0.6:
                scale_factor += 0.01
            elif normalized_position < 0.4:
                scale_factor -= 0.01
            scale_factor = max(0.5, min(2.0, scale_factor))

        if abs(flow_x_rot) > flow_threshold:
            rotation_angle += flow_x_rot * 5
        if abs(flow_x_trans) > flow_threshold:
            translation_x += flow_x_trans * 0.5  # Aumentar sensibilidad en X
        if abs(flow_y_trans) > flow_threshold:
            translation_y -= flow_y_trans * 0.5  # Aumentar sensibilidad en Y

        prev_gray = gray
        cv2.imshow("Cámara", frame)
        draw_scene()
        glfw.poll_events()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función principal
def main():
    global window
    if not glfw.init():
        sys.exit()

    window = glfw.create_window(800, 600, "Pueblo 3D con Transformaciones", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glViewport(0, 0, 800, 600)
    init()
    process_camera()
    glfw.terminate()

if __name__ == "__main__":
    main()