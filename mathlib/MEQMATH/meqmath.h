//---------------------------------------------------------------------------
#ifndef meqmathH
#define meqmathH

#include "matrix.h" 
#include "vector.h" 
#include "deqsys.h"

#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))  
/* Число пи деленное на 180 (для перевода градусов в радианы) */
extern const Extended M_PI_180;
/* 180 деленное на число пи (для перевода радиан в градусы) */
extern const Extended M_180_PI;
//---------------------------------------------------------------------------
/* Преобразование вектора в матрицу.
   Line=true - Результат матрица-строка, иначе матрица-столбец */
Matrix __fastcall VectorToMatrix(Vector V, bool Line = false);
/* Преобразование матрицы в вектор.
   Line=true - Результат матрица-строка, иначе матрица-столбец */
Vector __fastcall MatrixToVector(Matrix M);
/* Возвращает матрицу (3,3) поворота вокруг оси Ox (угол Angle - в рад) */
Matrix __fastcall RotX(Extended Angle);
/* Возвращает матрицу (3,3) поворота вокруг оси Oy (угол Angle - в рад) */
Matrix __fastcall RotY(Extended Angle);
/* Возвращает матрицу (3,3) поворота вокруг оси Oz (угол Angle - в рад) */
Matrix __fastcall RotZ(Extended Angle);
//---------------------------------------------------------------------------
#endif
