//*************************************************************************
//
//    Metodo de TV para removimiento de ruido en imagenes
//    
//
// Author       : Ivan de Jesus May-Cen
// Language     : C++
// Compiler     : g++
// Environment  : Linux Debian 11
// Revisions
//   Initial    : 2023-11-14 12:56:44 
//   Last       : 
//
//  para compilar
//    g++ -O2 mainROFimage.cpp -o test -lrt -lblitz `pkg-config --cflags opencv4` `pkg-config --libs opencv4`
//  para ejecutar
//    ./test archivoPNG LAMBDA
// 
//*************************************************************************

// preprocesor directives
#include <opencv2/core/core.hpp>                // OpenCV      
#include <opencv2/highgui/highgui.hpp>           
#include <blitz/array.h>                                // Blitz++             
#include <random/uniform.h>
#include <random/normal.h>
#include <sys/time.h>                   // funciones de tiempo
#include <cmath>                         // funciones matematicas
#include <float.h>                          // mathematical constants
#include <iostream>                                   


// declara namespace
using namespace std;
using namespace cv;
using namespace blitz;
using namespace ranlib;


// funciones a utilizar en la optimizacion
const double CTE = 0.2;
void Derivada(Array<double,2>&,Array<double,2>,Array<double,2>);
double minMod(double,double);                       
double Funcional(Array<double,2>,Array<double,2>);
void Print3D(Array<double,2>,FILE*,const char*);

// variables globales
double LAMBDA;
char imgname[50];

//*************************************************************************
//
//                        inicia funcion principal
//
//*************************************************************************
int main( int argc, char **argv )
{
  //parametros desde consola
  if(argc == 3) 
    {
     //version para lectura de imagen
     // read name of the file, read image
     strcpy( imgname, argv[1] );
     LAMBDA = atof(argv[2]);
    }
  // despliega informacion del proceso
  cout << endl << "Inicia procesamiento..." << endl << endl;

  // lectura de imagen
  Mat IMAGE = imread(imgname, IMREAD_GRAYSCALE); //GRAYSCALE
  
  // datos de la imagen
  int renglones = IMAGE.rows, columnas = IMAGE.cols;

  // inicializa funciones de ruido
  NormalUnit<double> ruidoInicial;              // Continuous normal distribution with mean = 0.0, variance = 1
  ruidoInicial.seed( (unsigned int)time(0) );
  Normal<double> ruidoImagen(0.0,CTE);    // Continuous normal distribution with mean = 0.0, variance = CTE
  ruidoImagen.seed( (unsigned int)time(0) );
  
  // separa memoria para procesar con Blitz
  Array<double,2> Is(renglones,columnas), Ic(renglones,columnas), Po(renglones,columnas), 
                           Gcx(renglones,columnas), Gcy(renglones,columnas),
                           Gsx(renglones,columnas), Gsy(renglones,columnas),
                           dummy(renglones,columnas);  
  Array<double,2> derivP(renglones,columnas), P(renglones,columnas), P0(renglones,columnas), P1(renglones,columnas), P2(renglones,columnas), Phase0(renglones,columnas);


  // datos de entrada
  double valMax = -DBL_MAX, valMin = DBL_MAX, num = 0.0, den = 0.0;
  for ( int r = 0; r < IMAGE.rows; r++ )
    for ( int c = 0; c < IMAGE.cols; c++ )
      {
        double ruido = ruidoImagen.random();
        Po(r,c) = 2.0 * double(IMAGE.at<uchar>(r,c))/255.0;
        // imagen ruidosa, se anade ruido aditivo
        Phase0(r,c) = Po(r,c) + 0.5*ruido;
        // valor inicial
        P(r,c) = Phase0(r,c);

        // calcula el SNR, mide ruido en la imagen
        num += ( P(r,c) * P(r,c) );
        den += ( (P(r,c)-Po(r,c)) * (P(r,c)-Po(r,c)) );
      }

  // despliega diferencia entre la estimacion y el valor real
  cout << endl << "SNR = " << 20.0*log10(num/den) << " db" << endl;

  // crea manejador de imagenes con openCV
  Mat Imagen( renglones, columnas, CV_64F, (unsigned char*) dummy.data() );

  const char *win0 = "Fase original";      namedWindow( win0, WINDOW_NORMAL );//AUTOSIZE 
  const char *win1 = "Estimaciones";      namedWindow( win1, WINDOW_NORMAL );//AUTOSIZE
  const char *win2 = "Imagen ruidosa";      namedWindow( win2, WINDOW_NORMAL );//AUTOSIZE

  dummy = Phase0; 
  imshow( win2, Imagen );
  imwrite("imagenRuidosa.png", 255*Imagen); //CV_LOAD_IMAGE_GRAYSCALE 

  // despliega valores de trabajo
  dummy = Po; 
  imshow( win0, Imagen );

  // ************************************************************************
  //             Inicia procesamiento
  // ************************************************************************
  struct timeval start, end;      // variables de manejo del tiempo
  gettimeofday( &start, NULL );    // marca tiempo de inicio
  
  // despliega valor inicial
  dummy = P;
  imshow( win1, Imagen );    waitKey( 1 );

  // inicia iteracion del algoritmo
  double Fx0, Fx = Funcional( P, Phase0 );
  double epsilon = 1.0e-8;         // criterio de paro del algoritmo
  double tao = 1.0e-3;             // tamaño de paso del descenso de gradiente
  unsigned iter = 0;               // iteraciones del algoritmo
  P2 = P;
  bool flag = true;
  while ( flag )
    {
      // guarda variables para seguimiento de la iteracion 
      P0 = P;    Fx0 = Fx;
      
      // estima la derivada de la funcional, usa descenso de gradiente 
      Derivada( derivP, P, Phase0 );

      // descenso de gradiente
      P = P - tao*derivP;

      Fx = Funcional( P, Phase0 );
      double difF = fabs(Fx0-Fx);



      // calcula error de la estimación, despliega avances
      //double errdp = sqrt( sum( pow2(derivP) ) );
      // calcula error de la estimación, despliega avances
      double errp = sqrt( sum( pow2(P-P0) ) ) / sqrt( sum( pow2(P0) ) );
      if ( (iter % 100) == 0 )
        {
          cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
          dummy = P;
          imshow( win1, Imagen );        waitKey( 1 );
        }
        
      // criterios de paro
      if ( (iter >= 10000) || (errp < epsilon) || ( difF < epsilon*errp) )
        {
          cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
          flag = false;
        }

      // incrementa contador iteracion
      iter++;
    }

  // termina funcion, calcula y despliega valores indicadores del proceso  
  gettimeofday( &end, NULL );    // marca de fin de tiempo cronometrado   

  // ************************************************************************
  //   resultados del procesamiento
  // ************************************************************************
  // calcula tiempo utilizado milisegundos
  double startms = double(start.tv_sec)*1000. + double(start.tv_usec)/1000.;
  double endms = double(end.tv_sec)*1000. + double(end.tv_usec)/1000.;
  double ms = endms - startms;
  cout << endl << "Tiempo empleado  : " << ms << " mili-segundos" << endl; 
  
  // despliega diferencia entre la estimacion y el valor real
  double error = sqrt( sum(pow2(Po-P)) ) / (sqrt( sum(pow2(Po)) ) + sqrt( sum(pow2(P)) ));
  cout << endl << "Normalized error : = " << error << endl << endl;
  dummy = log( 1.0 + fabs(Po-P) );
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imshow( win1, Imagen );        waitKey( 0 );
  dummy = P;
  imshow( win1, Imagen );        waitKey( 0 );

  imwrite("estimacionFinal.png", 255*Imagen); //CV_LOAD_IMAGE_GRAYSCALE
  dummy = fabs(P-Po);
  FILE* gnuplot_pipe = popen( "gnuplot -p", "w" );
  //Print3D( Po, gnuplot_pipe, "Original.eps" );
  //Print3D( P, gnuplot_pipe, "Estimada.eps" );
  Print3D( dummy, gnuplot_pipe, "Error.eps" );  
  pclose( gnuplot_pipe );  
  imwrite("Error.png", 255*Imagen); 

  // termina ejecucion del programa
  return 0;
}


//*************************************************************************
//
//    Funciones de trabajo
//
//*************************************************************************
// ************************************************************************
//       funcion principal de la derivada
//*************************************************************************
const double beta = 0.1;
const double eps = sqrt(DBL_EPSILON);
void Derivada( Array<double,2>& dP, Array<double,2> P, Array<double,2> Phase0 )
{
  // define parametro de regularizacion
  int columnas = P.cols();
  int renglones = P.rows();
  double lambda = LAMBDA;
  double V11x, V21x, V31x, 
         V12x, V22x, V32x,
         V11y, V21y, V31y, 
         V12y, V22y, V32y,
              Ux, Uy;

  // primera parte de la derivada
  //dP = lambda*( -(cos(P)-Ic)*sin(P) + (sin(P)-Is)*cos(P) );
  dP = lambda * ( P - Phase0 );
  
  // segunda parte de la derivada
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
        // procesa condiciones de frontera, eje x
        if ( r == renglones-1 )
          {  V11x = V21x = V31x = 0.0;  }
        else
          {
            Ux = P(r+1,c) - P(r,c);
            Uy = minMod( 0.5*(P(r+1,c+1) - P(r+1,c-1)), 0.5*(P(r,c+1) - P(r,c-1)) );
            V31x = Ux / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
        if ( r == 0 )
          {  V12x = V22x = V32x = 0.0;  }
        else
          {           
            Ux = P(r,c) - P(r-1,c);
            Uy = minMod( 0.5*(P(r,c+1) - P(r,c-1)), 0.5*(P(r-1,c+1) - P(r-1,c-1)) );
            V32x = Ux / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
      
        // procesa condiciones de frontera, eje y
        if ( c == columnas-1 )
          {  V11y = V21y = V31y = 0.0;  }
        else
          {          
            Ux = minMod( 0.5*(P(r+1,c+1) - P(r-1,c+1)), 0.5*(P(r+1,c) - P(r-1,c)) );
            Uy = P(r,c+1) - P(r,c);
            V31y = Uy / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
        if ( c == 0 )
          {  V12y = V22y = V32y = 0.0;  }
        else
          {
            Ux = minMod( 0.5*(P(r+1,c) - P(r-1,c)), 0.5*(P(r+1,c-1) - P(r-1,c-1)) );
            Uy = P(r,c) - P(r,c-1);
            V32y = Uy / sqrt( Ux*Ux + Uy*Uy + beta );
          }
          
        // actualiza valor de la derivada
        // termino divergencia
        dP(r,c) -= ( (V31x-V32x) + (V31y-V32y) );   
      }
}

// ************************************************************************
//       funcional
//*************************************************************************
double Funcional( Array<double,2> P, Array<double,2> Phase0 )
{
  // define parametro de regularizacion
  int columnas = P.cols();
  int renglones = P.rows();
  double lambda = LAMBDA;
  double hx = 1.0 / (double(renglones)-1.0);
  double hy = 1.0 / (double(columnas)-1.0);
  
  
  // evalua la funcional 
  double dx, dy, dxc, dxs, dyc, dys, v1, v2, vxc, vxs, vyc, vys; 
  double suma = 0.0;
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
        // evalua derivadas, en x
        if ( r == renglones-1 )
          {  dx = P(r,c) - P(r-1,c); }
        else if ( r == 0 )
          {  dx = P(r+1,c) - P(r,c); }
        else
          {  dx = 0.5*(P(r+1,c) - P(r-1,c));  }

        // evalua derivadas, en y
        if ( c == columnas-1 )
          {  dy = P(r,c) - P(r,c-1); }
        else if ( c == 0 )
          {  dy = P(r,c+1) - P(r,c);  }
        else
          {  dy = 0.5*(P(r,c+1) - P(r,c-1));  }

        // evalua terminos de similitud 
        v1 = P(r,c) - Phase0(r,c);
        
        // evalua funcional en (r,c)  
        // con termino de ajuste y variacion total      
        suma += 0.5*lambda*( v1*v1 ) + sqrt(dx*dx+dy*dy);
      }

  // regresa valor
  return suma*hx*hy;
}

//*************************************************************************
//
//    Funciones para despliegue de resultados
//
//*************************************************************************
void Print3D( Array<double,2> Z, FILE* salida, const char* fileName )
{
  // salida en modo grafico o en archivo
  fprintf( salida, "set terminal postscript eps enhanced rounded\n");
  fprintf( salida, "set output \"%s\"\n", fileName );  
  fprintf( salida, "set style line 1 linetype -1 linewidth 1\n" );
  fprintf( salida, "set xlabel \"columns (pixels)\" offset -1,-1\n" );
  fprintf( salida, "set ylabel \"rows (pixels)\" offset -1,-1\n" );
  fprintf( salida, "set zlabel \"phase\"\n" );
  fprintf( salida, "set xrange [%f:%f]\n", 0., float(Z.cols()) );
  fprintf( salida, "set yrange [%f:%f]\n", 0., float(Z.rows()) );
//  fprintf( salida, "set zrange [%f:%f]\n", -4.0, 4.0 );
  fprintf( salida, "set xtics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set ytics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set view 70, 210\n" );
  fprintf( salida, "unset key\n" );
  fprintf( salida, "unset colorbox\n" );
  fprintf( salida, "set hidden3d front\n" );
//  fprintf( salida, "set xyplane at -45.0\n" );
  fprintf( salida, "splot '-' using 1:2:3 title '' with lines lt -1 lw 0.1\n" );
  for ( int c = 0; c < Z.cols(); c += 8 )
    {
      for ( int r = 0; r < Z.rows(); r += 8 )
        fprintf( salida, "%f %f %f\n", float(c), float(r), float(Z(r,c)) );
      fprintf(  salida, "\n" );      // New row (datablock) separated by blank record
    }
  fprintf( salida, "e\n" );
  fflush( salida );
}



// ***************************************************************
//   Condiciones de frontera Neumann
// ***************************************************************
void boundaryCond1( Array<double,2>& T )
{
  // define parametro de regularazacion
  int columnas = T.cols();
  int renglones = T.rows();
  blitz::Range all = blitz::Range::all();

  // condiciones de frontera
  T(0,all) = T(1,all);
  T(renglones-1,all) = T(renglones-2,all);
  T(all,0) = T(all,1);
  T(all,columnas-1) = T(all,columnas-2);
  T(0,0) = T(1,1);
  T(0,columnas-1) = T(1,columnas-2);
  T(renglones-1,0) = T(renglones-2,1);
  T(renglones-1,columnas-1) = T(renglones-2,columnas-2);
}
// ***************************************************************
//   min-mod
// ***************************************************************
double minMod( double a, double b )
{
  // minmod operator
  double signa = (a > 0.0) ? 1.0 : ((a < 0.0) ? -1.0 : 0.0);
  double signb = (b > 0.0) ? 1.0 : ((b < 0.0) ? -1.0 : 0.0);
//  double minim = fmin( fabs(a), fabs(b) ); 
  double minim = ( fabs(a) <= fabs(b) ) ? fabs(a) : fabs(b); 
  return ( (signa+signb)*minim/2.0 );

  // geometric average
//  return( 0.5*(a+b) ); Total Variation Diminishing Runge-Kutta Schemes
  
  // upwind 
//  double maxa = (a > 0.0) ? a : 0.0;
//  double maxb = (b > 0.0) ? b : 0.0;
//  return( 0.5*(maxa+maxb) );  
}




