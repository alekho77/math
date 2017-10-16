/*
    Рашение уравнений и систем уравнений
    08.01.2000
    (c) Хозин Алексей Ю. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#include <vcl.h>
#include <math.h>
#pragma hdrstop

#include "deqsys.h"
#define MaxDeqSysErrCode    6
//---------------------------------------------------------------------------
#pragma package(smart_init)
//---------------------------------------------------------------------------
/*  Приведение матрицы коэффициентов к верхней треугольной, совместно с матрицей
    свободных коэффициентов. A и B изменяються соответственно. Возврешает матрицу A */
Matrix& __fastcall UpTrnMat_qs(Matrix& A, Matrix& B, int& iErr)
{
if(A.Lin!=A.Col)
  {
  iErr=1; // Матрица коэффициентов не квадратная
  return A;
  }
if(B.Col!=1)
  {
  iErr=2; // Число столбцов в матрице свободных коэффициентов не равно 1
  return A;
  }
if(A.Lin!=B.Lin)
  {
  iErr=3; // Числа строк в матрицах коэффициентов не равны
  return A;
  }
Extended Fact, Amax, Atmp, Btmp;
int i, j, k, lm;
for(i=1;i<A.Lin;i++)
  {
  /* Определяем максимальное число в i-ом столбце в строках ниже i-ой */
  Amax=A[A.Index(i,i)];
  lm=i;
  for(j=i+1;j<=A.Lin;j++)
    if( fabsl(Amax) < fabsl(A[A.Index(j,i)]) )
      {
      lm=j;
      Amax=A[A.Index(j,i)];
      }
  if(Amax==0.0L)continue;
  /* Если необходимо, то обмен местами строк */
  if(i!=lm)
    {
    for(j=i;j<=A.Col;j++)
      {
      Atmp=A[A.Index(i,j)];
      A[A.Index(i,j)]=A[A.Index(lm,j)];
      A[A.Index(lm,j)]=Atmp;
      }
    Btmp=B[i];
    B[i]=B[lm];
    B[lm]=Btmp;
    }
  /* Перечисление коэффициентов в строках ниже i-ой */
  for(j=i+1;j<=A.Lin;j++)
    {
    if(A[A.Index(j,i)]==0.0L) continue; // строка не перечисляется
    Fact=-A[A.Index(j,i)]/Amax;
    A[A.Index(j,i)]=0.0L; // На самом деле эту операцию можно опустить
    for(k=i+1;k<=A.Col;k++)
      A[A.Index(j,k)]+=A[A.Index(i,k)]*Fact;
    B[j]+=B[i]*Fact;
    }
  }
iErr=0;
return A;
}
//---------------------------------------------------------------------------
Extended __fastcall Det_qs(Matrix& A)
{
Extended rez=1.0L;
int i;
for(i=1;i<=A.Lin;i++)
  rez*=A[(i-1)*A.Col+i];
return rez;
}
//---------------------------------------------------------------------------
Matrix __fastcall DeqSys(const Matrix& A, const Matrix& B, int& iErr)
{
Matrix A1(A),B1(B),X;
int i,j;
/* Прямой ход */
if(MatrixError->LastErr)
  {
  iErr=4; // Ошибка при инициализации временных матриц
  return X;
  }
if(fabsl(Det_qs(UpTrnMat_qs(A1,B1,iErr)))==0.0)
  {
  iErr=5; // Детерминант матрицы коэффициентов равен нулю
  return X;
  }
if(iErr)return X;
/* Обратный ход */
X=B1;
for(i=A1.Lin;i>0;i--)
  {
  for(j=A1.Col;j>i;j--)
    X[i]-=X[j]*A1[(i-1)*A1.Col+j];
  X[i]/=A1[(i-1)*A1.Col+i];
  }
iErr=0;
return X;
}
//---------------------------------------------------------------------------
Matrix __fastcall Inex(const Matrix& A, const Matrix& X, const Matrix& B)
{
Matrix I;
I=B-A*X;
return I;
}
//---------------------------------------------------------------------------
Extended __fastcall SqrtInex(const Matrix& A, const Matrix& X, const Matrix& B)
{
Matrix I,TI;
I=Inex(A,X,B);
I=Trans(I)*I;
return sqrtl(I[1]);
}
//---------------------------------------------------------------------------
Matrix __fastcall Trans(Matrix& M)
{
Matrix T(M.Col,M.Lin);
int i,j;
for(i=1;i<=M.Lin;i++)
  for(j=1;j<=M.Col;j++)
    T[(j-1)*T.Col+i]=M[(i-1)*M.Col+j];
return T;
}
//---------------------------------------------------------------------------
/* Получить строку с описанием ошибки ErrCode */
AnsiString __fastcall DeqSysErrStr(int ErrCode)
{
char *dseStr[MaxDeqSysErrCode]={
/* 0*/ "DeqSys : Ошибок нет.",
/* 1*/ "UpTrnMat_qs() : Матрица коэффициентов не квадратная.",
/* 2*/ "UpTrnMat_qs() : Число столбцов в матрице свободных коэффициентов не равно 1.",
/* 3*/ "UpTrnMat_qs() : Числа строк в матрицах коэффициентов не равны.",
/* 4*/ "DeqSys() : Ошибка при инициализации временных матриц.",
/* 5*/ "DeqSys() : Детерминант матрицы коэффициентов равен нулю."
};
if((ErrCode<0)||(ErrCode>=MaxDeqSysErrCode))
  return AnsiString("DeqSysErrStr() : Неизвестный код ошибки");
return AnsiString(dseStr[ErrCode]);
}
//---------------------------------------------------------------------------
