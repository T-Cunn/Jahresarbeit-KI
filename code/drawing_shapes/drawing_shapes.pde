float x = 0;
float y = 0;
float z = 0;
float xs = 0;
float ys = 0;
float zs = 0;
float sizeMin = 6;
float sizeMax = 14;
float size = sizeMin;

int fuzzCount = 0;
int[] fuzzX = new int[fuzzCount];
int[] fuzzY = new int[fuzzCount];

int c = 0;
int total = 1000;

void setup() {
  size(28, 28, P3D);
  for (int i=0; i<fuzzCount; i++) {
    fuzzX[i] = floor(random(28));
    fuzzY[i] = floor(random(28));
  }
}

void draw() {
  background(0);
  translate(14, 14, 0);
  rotateX(x);
  rotateY(y);
  rotateZ(z);
  noStroke();
  fill(255);
  //box(size); //6-14
  //tetrahedron(size); //4-8
  Octahedron(size); //6-14
  xs=(xs+random(-0.01, 0.01)*0.99);
  ys=(ys+random(-0.01, 0.01)*0.99);
  zs=(zs+random(-0.01, 0.01)*0.99);
  x+=xs;
  y+=ys;
  z+=zs;
  size+=(sizeMax-sizeMin)/total;
  camera();
  hint(DISABLE_DEPTH_TEST);
  for (int i=0; i<fuzzCount; i++) {
    if (random(1)>0.5) fill(255); else fill(0);
    rect(fuzzX[i], fuzzY[i], 1, 1);
    fuzzX[i] = floor(random(28));
    fuzzY[i] = floor(random(28));
  }
  saveFrame("octahedron/####.png");
  c++;
  if (c==total) exit();
}

//Code from https://discourse.processing.org/t/how-to-make-crystal-like-shapes-with-processing/10392/6
void tetrahedron(float size) {
  scale (size);
  beginShape();
  vertex(1, 1, -1);
  vertex(-1, -1, -1);
  vertex(-1, 1, 1) ;
  endShape();
  beginShape();
  vertex(1, -1, 1);
  vertex(-1, -1, -1);
  vertex(-1, 1, 1) ;
  endShape();
  beginShape();
  vertex(1, -1, 1);
  vertex(-1, -1, -1);
  vertex(1, 1, -1);
  endShape();
  beginShape();
  vertex(1, 1, -1);
  vertex(1, -1, 1);
  vertex(-1, 1, 1) ;
  endShape();
}
void vertexPVector ( PVector pos ) {
  vertex(pos.x, pos.y, pos.z);
}
void makeShape( PVector p1, PVector p2, PVector p3 ) {
  beginShape();
  vertexPVector(p1);
  vertexPVector(p2);
  vertexPVector(p3);
  endShape();
}
void Octahedron(float size) {
  // define vertices
  PVector [] list = new PVector[6];
  list [0] = new PVector  (1, 0, 0);
  list [1] = new PVector  (0, 0, 1);
  list [2] = new PVector  (0, -1, 0);
  list [3] = new PVector  (-1, 0, 0);
  list [4] = new PVector  (0, 0, -1);
  list [5] = new PVector  (0, 1, 0);
  for (int i = 0; i < list.length; i++) {
    list[i].mult(size);
  }
  // defines the shapes
  // the lower half
  makeShape (list [0], list [1], list [5] ) ;
  makeShape (list [0], list [4], list [5] ) ;
  makeShape (list [3], list [4], list [5] ) ;
  makeShape (list [3], list [1], list [5] ) ;
  // the upper half
  makeShape (list [0], list [1], list [2] ) ;
  makeShape (list [0], list [4], list [2] ) ;
  makeShape (list [3], list [4], list [2] ) ;
  makeShape (list [3], list [1], list [2] ) ;
}
