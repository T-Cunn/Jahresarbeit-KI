#!/usr/bin/env python3

import csv
import numpy as np
import numpy.random as npr
from turtle import *

#setup screen
width, height = 200, 200
screen = Screen()
screen.setup(width, height)
screen.tracer(0, 0)

class ball:
  """ball class"""
  
  #size of the circle
  diameter = 20
  
  def __init__(self, x, y, vx, vy):
    self.pos = np.array([x, y]) # pos = x, y vector    
    self.vel = np.array([vx, vy]) # vel = x, y velocity vector
    self.newVel = self.vel # velocity after a collision
    
    self.turtle = Turtle() # turtle used for drawing
    self.turtle.shape("circle")
    self.turtle.shapesize(self.diameter/20, self.diameter/20, 0)
    self.turtle.penup()
    self.turtle.speed(0)
    
  def update_pos(self):
    self.vel = self.newVel
    self.pos = self.pos + self.vel
  
  def draw(self):
    self.turtle.goto(self.pos[0], self.pos[1])
    
  def check_wall_collisions(self):
    if self.pos[0]+self.diameter/2 > (width/2):
      self.vel[0] *= -1
    if self.pos[0]-self.diameter/2 < -(width/2):
      self.vel[0] *= -1
    if self.pos[1]+self.diameter/2 > (height/2):
      self.vel[1] *= -1
    if self.pos[1]-self.diameter/2 < -(height/2):
      self.vel[1] *= -1
  
  def check_other_collisions(self, others):
    for other in others:
      if other != self:
        difference = other.pos-self.pos
        dist = (difference**2).sum()**0.5
        if abs(dist) < self.diameter/2 + other.diameter/2: # if collision
          # calculate rotation matrix
          difference_unit = difference / dist
          deg90 = np.array([[0, -1], [1, 0]])
          difference_unit90 = np.matmul(deg90, difference_unit)
          back_rotation_matrix = np.flip(np.array([difference_unit, difference_unit90]))
          rotation_matrix = np.linalg.inv(back_rotation_matrix)
          # calculate new velocity
          selfRotVel = np.matmul(rotation_matrix, self.vel)
          otherRotVel = np.matmul(rotation_matrix, other.vel)
          newRotVel = np.array([otherRotVel[0], selfRotVel[1]])
          self.newVel = np.matmul(back_rotation_matrix, newRotVel)*1.01
          
def update_screen():
  #e = 0
  for ball in balls:
    ball.check_wall_collisions()
    ball.check_other_collisions(balls)
    #e += (ball.vel**2).sum()
    
  for ball in balls:
    ball.update_pos()
    ball.draw()
    
  #screen.update()
  #print(e)
  #screen.ontimer(update_screen, 5)


X = open("X.csv", 'w')
csvwriter_X = csv.writer(X)
y = open("y.csv", 'w')
csvwriter_y = csv.writer(y)
  
for i in range(10000):
  print(i)
  
  #define 3 balls
  balls = []
  balls.append(ball(0, 0, npr.randint(-100, 100)/100, npr.randint(-100, 100)/100))
  #balls.append(ball(width/4, -height/4, npr.randint(-100, 100)/100, npr.randint(-100, 100)/100))
  #balls.append(ball(0, height/4, npr.randint(-100, 100)/100, npr.randint(-100, 100)/100))
  
  #save ball positions
  printer = []
  for b in balls:
    printer.append(b.pos[0])
    printer.append(b.pos[1])
    printer.append(b.vel[0])
    printer.append(b.vel[1])
  csvwriter_X.writerow(printer)
  
  #run 100 iterations
  for j in range(50):
    update_screen()
  
  #save end positions
  printer = []
  for b in balls:
    printer.append(b.pos[0])
    printer.append(b.pos[1])
    printer.append(b.vel[0])
    printer.append(b.vel[1])
  csvwriter_y.writerow(printer)

X.close()
y.close()

#update_screen()
#screen.exitonclick()
