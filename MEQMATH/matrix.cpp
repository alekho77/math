/*
    Описание объекта типа "Матрица"
    08.01.2000
    (c) Хозин Алексей Ю. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#include <vcl.h>
#include <stdio.h>
#pragma hdrstop

#include "matrix.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

#define MaxMatrixErrCode    10
TMatrixError *MatrixError = new TMatrixError;
//---------------------------------------------------------------------------
        /* Высвобождение памяти, Col и Lin не изменяются */
void __fastcall Matrix::Free(void)
{
if(M!=NULL)delete M;
M=NULL;
}
//---------------------------------------------------------------------------
        /* Если Col и Lin не равны нулю, создает массив (Col*Lin) чисел */
void __fastcall Matrix::TryCreate(void)
{
if(Col&&Lin)
  {
  M=new Extended[Col*Lin];
  if(M==NULL)MatrixError->Add(1);
  }
}
//---------------------------------------------------------------------------
void __fastcall Matrix::SetCol(const int Val)
{
if((Val>=0)&&(FCol!=Val))
  {
  Free();
  FCol=Val;
  TryCreate();
  }
}
//---------------------------------------------------------------------------
void __fastcall Matrix::SetLin(const int Val)
{
if((Val>=0)&&(FLin!=Val))
  {
  Free();
  FLin=Val;
  TryCreate();
  }
}
//---------------------------------------------------------------------------
__fastcall Matrix::Matrix()
{
FCol=FLin=0;
M=NULL;
}
//---------------------------------------------------------------------------
__fastcall Matrix::Matrix(const int L, const int C)
{
FCol=FLin=0;
M=NULL;
Lin=L;
Col=C;
}
//---------------------------------------------------------------------------
__fastcall Matrix::Matrix(const Matrix& MSrc)
{
FCol=FLin=0;
M=NULL;
*this=MSrc;
}
//---------------------------------------------------------------------------
__fastcall Matrix::~Matrix()
{
Free();
}
//---------------------------------------------------------------------------
    /* Создание копии (предыдущая информация в матрице теряется) */
    /* Если матрица источник пуста, то получаемая матрица будет пустой */
Matrix& __fastcall Matrix::operator = (const Matrix& MSrc)
{
Col=MSrc.Col;
Lin=MSrc.Lin;
int i;
if(MSrc.M!=NULL)
  for(i=0;i<Lin*Col;i++) M[i]=MSrc.M[i];
return *this;
}
//---------------------------------------------------------------------------
    /* Инициализация матрицы массивом чисел.
       Если Src==NULL высвобождение занимаемой памяти, и Lin=Col=0.
       Необходимо зарание задать размеры. */
Matrix& __fastcall Matrix::operator = (const Extended *Src)
{
if(Src==NULL)
  {
  Free();
  FCol=FLin=0;
  return *this;
  }
if(!(Col&&Lin))
  {
  MatrixError->Add(2); // Размер матрицы не определен
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]=Src[i];
return *this;
}
//---------------------------------------------------------------------------
    /* Инициализация числом. Необходимо зарание задать размеры */
Matrix& __fastcall Matrix::operator = (const Extended Initial)
{
if(!(Col&&Lin))
  {
  MatrixError->Add(2); // Размер матрицы не определен
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]=Initial;
return *this;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator += (const Matrix& MSum)
{
if((M==NULL)||(MSum.M==NULL))
  {
  MatrixError->Add(3); // Матрица слагаемое не определена
  return *this;
  }
if((Lin!=MSum.Lin)||(Col!=MSum.Col))
  {
  MatrixError->Add(4); // Несоответствие между размерами матриц
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]+=MSum.M[i];
return *this;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator + (const Matrix& MS) const
{
Matrix MSum(*this);
MSum+=MS;
return MSum;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator -= (const Matrix& MSum)
{
if((M==NULL)||(MSum.M==NULL))
  {
  MatrixError->Add(3); // Матрица слагаемое не определена
  return *this;
  }
if((Lin!=MSum.Lin)||(Col!=MSum.Col))
  {
  MatrixError->Add(4); // Несоответствие между размерами матриц
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]-=MSum.M[i];
return *this;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator - (const Matrix& MS) const
{
Matrix MSum(*this);
MSum-=MS;
return MSum;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator *= (const Extended& Val)
{
if(M==NULL)
  {
  MatrixError->Add(5); // Матрица множитель не определена
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]*=Val;
return *this;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator /= (const Extended& Val)
{
if(M==NULL)
  {
  MatrixError->Add(5); // Матрица множитель не определена
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]/=Val;
return *this;
}
//---------------------------------------------------------------------------
Matrix __fastcall operator * (const Extended& Val, const Matrix& MMul)
{
Matrix MRez(MMul);
MRez*=Val;
return MRez;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator * (const Extended& Val) const
{
Matrix MRez(*this);
MRez*=Val;
return MRez;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator / (const Extended& Val) const
{
Matrix MRez(*this);
MRez/=Val;
return MRez;
}
//---------------------------------------------------------------------------
    /* Размерности матриц : M1(l,m)*M2(m,n)=M3(l,n) */
Matrix& __fastcall Matrix::operator *= (const Matrix& MMul)
{
(*this)=(*this)*MMul;
return *this;
}
//---------------------------------------------------------------------------
    /* Размерности матриц : M1(l,m)*M2(m,n)=M3(l,n) */
Matrix __fastcall Matrix::operator * (const Matrix& MMul) const
{
if((M==NULL)||(MMul.M==NULL))
  {
  MatrixError->Add(5); // Матрица множитель не определена
  return *this;
  }
if(Col!=MMul.Lin)
  {
  MatrixError->Add(6); // Несоответствие между размерами матриц множителей
  return *this;
  }
Matrix MRez(Lin,MMul.Col);
int i,j,k;
for(i=0;i<Lin;i++)
  for(j=0;j<MMul.Col;j++)
    {
    MRez.M[i*MMul.Col+j]=0.0L;
    for(k=0;k<Col;k++)
      MRez.M[i*MMul.Col+j]+=M[i*Col+k]*MMul.M[k*MMul.Col+j];
    }
return MRez;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator - (void) const
{
if(M==NULL)
  {
  MatrixError->Add(7); // Матрица не определена
  return *this;
  }
Matrix MRez(*this);
int i;
for(i=0;i<Lin*Col;i++) MRez.M[i]=-MRez.M[i];
return MRez;
}
//---------------------------------------------------------------------------
    /* Равенство только если :
        Резмеры совпадают и равны между собой соответствующие элементы */
bool __fastcall Matrix::operator == (const Matrix& MCom) const
{
if((M==NULL)||(MCom.M==NULL))
  {
  MatrixError->Add(8); // Матрицы не определены
  return false;
  }
if((Col!=MCom.Col)||(Lin!=MCom.Lin)) return false;
bool Rez;
int i;
for(i=0;i<Lin*Col;i++)
  {
  Rez=(M[i]==MCom.M[i]);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Matrix::operator != (const Matrix& MCom) const
{
return !((*this)==MCom);
}
//---------------------------------------------------------------------------
        /* Равенство только, если : Все элементы равны заданному числу */
bool __fastcall Matrix::operator == (const Extended& Val) const
{
if(M==NULL)
  {
  MatrixError->Add(8); // Матрицы не определены
  return false;
  }
bool Rez;
int i;
for(i=0;i<Lin*Col;i++)
  {
  Rez=(M[i]==Val);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Matrix::operator != (const Extended& Val) const
{
return !((*this)==Val);
}
//---------------------------------------------------------------------------
    /* Индексный оператор. Count = 1..(Lin*Col) */
Extended& __fastcall Matrix::operator [] (int Count)
{
if(Count<=0)
  {
  MatrixError->Add(9); // Некорректно заданный индекс
  Count=1;
  }
else if(Count>Lin*Col)
       {
       MatrixError->Add(9); // Некорректно заданный индекс
       Count=Lin*Col;
       }
return M[Count-1];
}
//---------------------------------------------------------------------------
    /* Фукция для определения индкса элемента по номеру строки и столбца */
int __fastcall Matrix::Index(int L, int C)
{
return (L-1)*Col+C;
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
    /* Возвращает 0, если массив пуст */
int __fastcall TMatrixError::GetLastError(void)
{
if(Count)return iErr[Count-1];
return Count;
}
//---------------------------------------------------------------------------
__fastcall TMatrixError::TMatrixError()
{
FCount=0;
}
//---------------------------------------------------------------------------
__fastcall TMatrixError::~TMatrixError()
{
if(Count)delete iErr;
}
//---------------------------------------------------------------------------
    /* Можно получить строку с описанием ошибки, код которой находится в массиве под номером Index */
    /* Index первого элемента = 1 */
char *MatrixErrStr[MaxMatrixErrCode]={
/* 0*/ "TMatrixError::ErrStr() : Вы неправильно указали индекс.",
/* 1*/ "Matrix::Create() : Не хватает памяти для матрицы.",
/* 2*/ "Matrix::operator = : Инициализация невозможна, так как размер матрицы не определен.",
/* 3*/ "Matrix::operator +(-) : Матрица слагаемое не определена.",
/* 4*/ "Matrix::operator +(-) : Несоответствие между размерами матриц слагаемых."
/* 5*/ "Matrix::operator *(/) : Матрица множитель не определена.",
/* 6*/ "Matrix::operator * : Несоответствие между размерами матриц множителей.",
/* 7*/ "Унарный Matrix::operator - : Матрица не определена.",
/* 8*/ "Matrix::operator == : Матрицы не определены.",
/* 9*/ "Matrix::operator [] : Некорректно заданный индекс"
};
AnsiString __fastcall TMatrixError::ErrStr(int Index)
{
if((Index<=0)||(Index>Count))Index=0;
else Index=iErr[Index-1];
return AnsiString(MatrixErrStr[Index]);
}
//---------------------------------------------------------------------------
    /* Очишает массив с ошибками */
void __fastcall TMatrixError::Clear(void)
{
if(Count)delete iErr;
FCount=0;
}
//---------------------------------------------------------------------------
    /* Добавить ошибку */
void __fastcall TMatrixError::Add(int ErrCode)
{
if((ErrCode<1)||(ErrCode>=MaxMatrixErrCode))return;
int *iErrt=new int[Count+1];
int i;
for(i=0;i<Count;i++)iErrt[i]=iErr[i];
if(Count)delete iErr;
iErr=iErrt;
iErr[Count]=ErrCode;
FCount++;
}
//---------------------------------------------------------------------------

