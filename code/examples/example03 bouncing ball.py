import turtle

turtle.shape("circle")
turtle.penup()

x, y = 0, 0
xdir, ydir = 3, 3
xlimit, ylimit = turtle.window_width() / 2, turtle.window_height() / 2

def move():
    global x, y, xdir, ydir

    x = x + xdir
    y = y + ydir

    if not -xlimit < x < xlimit:
        xdir = -xdir
    if not -ylimit < y < ylimit:
        ydir = -ydir

    turtle.goto(x, y)

    turtle.ontimer(move, 5)

turtle.ontimer(move, 5)

turtle.exitonclick()
