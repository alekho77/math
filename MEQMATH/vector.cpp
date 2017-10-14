/*
    Описание объекта типа "Вектор"
    08.01.2000
    (c) Хозин Алексей Ю. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#include <vcl.h>
#include <math.h>
#pragma hdrstop

#include "vector.h"
#include "matrix.h"
Matrix __fastcall VectorToMatrix(Vector V, bool Line = false);
Vector __fastcall MatrixToVector(Matrix M);
//---------------------------------------------------------------------------
#pragma package(smart_init)

#define MaxVectorErrCode    10
TVectorError *VectorError = new TVectorError;
//---------------------------------------------------------------------------
/*  Высвобождение памяти, Dim не изменяется */
void __fastcall Vector::Free(void)
{
if(V!=NULL)delete V;
V=NULL;
}
//---------------------------------------------------------------------------
/* Если Dim не равен нулю, создает массив Dim чисел */
void __fastcall Vector::TryCreate(void)
{
if(Dim)
  {
  V=new Extended[Dim];
  if(V==NULL)VectorError->Add(1); // Нехватка памяти
  }
}
//---------------------------------------------------------------------------
void __fastcall Vector::SetDim(const int Val)
{
if((Val>=0)&&(FDim!=Val))
  {
  Free();
  FDim=Val;
  TryCreate();
  }
}
//---------------------------------------------------------------------------
__fastcall Vector::Vector()
{
FDim=0;
V=NULL;
}
//---------------------------------------------------------------------------
__fastcall Vector::Vector(const int D)
{
FDim=0;
V=NULL;
Dim=D;
}
//---------------------------------------------------------------------------
__fastcall Vector::Vector(const Vector& VSrc)
{
FDim=0;
V=NULL;
*this=VSrc;
}
//---------------------------------------------------------------------------
__fastcall Vector::~Vector()
{
Free();
}
//---------------------------------------------------------------------------
    /* Создание копии (предыдущая информация в векторе теряется) */
Vector& __fastcall Vector::operator = (const Vector& VSrc)
{
Dim=VSrc.Dim;
int i;
if(VSrc.V!=NULL)
  for(i=0;i<Dim;i++) V[i]=VSrc.V[i];
return *this;
}
//---------------------------------------------------------------------------
    /* Инициализация вектора массивом чисел
       (если Src==NULL высвобождение занимаемой памяти, и Dim=0)
       Необходимо зарание задать размер */
Vector& __fastcall Vector::operator = (const Extended *Src)
{
if(Src==NULL)
  {
  Free();
  FDim=0;
  return *this;
  }
if(!Dim)
  {
  VectorError->Add(2); // Размерность вектора не определена
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]=Src[i];
return *this;
}
//---------------------------------------------------------------------------
    /* Необходимо зарание задать размер */
Vector& __fastcall Vector::operator = (const Extended Initial)
{
if(!Dim)
  {
  VectorError->Add(2); // Размерность вектора не определена
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]=Initial;
return *this;
}
//---------------------------------------------------------------------------
    /* Размерности векторов должны совпадать */
Vector& __fastcall Vector::operator += (const Vector& VSum)
{
if((V==NULL)||(VSum.V==NULL))
  {
  VectorError->Add(3); // Вектор слагаемое не определен
  return *this;
  }
if(Dim!=VSum.Dim)
  {
  VectorError->Add(4); // Несоответствие между размерами векторов
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]+=VSum.V[i];
return *this;
}
//---------------------------------------------------------------------------
    /* Размерности векторов должны совпадать */
Vector __fastcall Vector::operator + (const Vector& VS) const
{
Vector VSum(*this);
VSum+=VS;
return VSum;
}
//---------------------------------------------------------------------------
    /* Размерности векторов должны совпадать */
Vector& __fastcall Vector::operator -= (const Vector& VSum)
{
if((V==NULL)||(VSum.V==NULL))
  {
  VectorError->Add(3); // Вектор слагаемое не определен
  return *this;
  }
if(Dim!=VSum.Dim)
  {
  VectorError->Add(4); // Несоответствие между размерами векторов
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]-=VSum.V[i];
return *this;
}
//---------------------------------------------------------------------------
    /* Размерности векторов должны совпадать */
Vector __fastcall Vector::operator - (const Vector& VS) const
{
Vector VSum(*this);
VSum-=VS;
return VSum;
}
//---------------------------------------------------------------------------
Vector& __fastcall Vector::operator *= (const Extended& Val)
{
if(V==NULL)
  {
  VectorError->Add(5); // Вектор множитель не определен
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]*=Val;
return *this;
}
//---------------------------------------------------------------------------
Vector& __fastcall Vector::operator /= (const Extended& Val)
{
if(V==NULL)
  {
  VectorError->Add(5); // Вектор множитель не определен
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]/=Val;
return *this;
}
//---------------------------------------------------------------------------
Vector __fastcall operator * (const Extended& Val, const Vector& VMul)
{
Vector VRez(VMul);
VRez*=Val;
return VRez;
}
//---------------------------------------------------------------------------
Vector __fastcall Vector::operator * (const Extended& Val) const
{
Vector VRez(*this);
VRez*=Val;
return VRez;
}
//---------------------------------------------------------------------------
Vector __fastcall Vector::operator / (const Extended& Val) const
{
Vector VRez(*this);
VRez/=Val;
return VRez;
}
//---------------------------------------------------------------------------
    /* Скалярное умножение. Размерности векторов должны совпадать. */
Extended __fastcall Vector::operator * (const Vector& VMul) const
{
if((V==NULL)||(VMul.V==NULL))
  {
  VectorError->Add(5); // Вектор множитель не определен
  return 0.0L;
  }
if(Dim!=VMul.Dim)
  {
  VectorError->Add(6); // Несоответствие между размерами векторов множителей
  return 0.0L;
  }
Extended Rez=0.0L;
int i;
for(i=0;i<Dim;i++)
  Rez+=V[i]*VMul.V[i];
return Rez;
}
//---------------------------------------------------------------------------
    /* Умножение матрицы на вектор. Размеры : M(m,n)*V(n) = V(m) */
Vector __fastcall operator * (const Matrix& M, const Vector& V)
{
return MatrixToVector(M*VectorToMatrix(V));
}
//---------------------------------------------------------------------------
Vector __fastcall Vector::operator - (void) const
{
if(V==NULL)
  {
  VectorError->Add(7); // Вектор не определена
  return *this;
  }
Vector VRez(Dim);
int i;
for(i=0;i<Dim;i++) VRez.V[i]=-V[i];
return VRez;
}
//---------------------------------------------------------------------------
    /* Операторы сравнения */
    /* Равенство только, если : Резмеры совпадают и равны между собой соответствующие элементы */
bool __fastcall Vector::operator == (const Vector& VCom) const
{
if((V==NULL)||(VCom.V==NULL))
  {
  VectorError->Add(8); // Векторы не определены
  return false;
  }
if(Dim!=VCom.Dim) return false;
bool Rez;
int i;
for(i=0;i<Dim;i++)
  {
  Rez=(V[i]==VCom.V[i]);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Vector::operator != (const Vector& VCom) const
{
return !((*this)==VCom);
}
//---------------------------------------------------------------------------
        /* Равенство только, если : Все элементы равны заданному числу */
bool __fastcall Vector::operator == (const Extended& Val) const
{
if(V==NULL)
  {
  VectorError->Add(8); // Векторы не определены
  return false;
  }
bool Rez;
int i;
for(i=0;i<Dim;i++)
  {
  Rez=(V[i]==Val);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Vector::operator != (const Extended& Val) const
{
return !((*this)==Val);
}
//---------------------------------------------------------------------------
    /* Индексный оператор. Count = 1..Dim */
Extended& __fastcall Vector::operator [] (int Count)
{
if(Count<=0)
  {
  VectorError->Add(9); // Некорректно заданный индекс
  Count=1;
  }
else if(Count>Dim)
       {
       VectorError->Add(9); // Некорректно заданный индекс
       Count=Dim;
       }
return V[Count-1];
}
//---------------------------------------------------------------------------
    /* Определение модуля вектора */
Extended __fastcall Vector::Mod(void)
{
Extended Rez;
Rez=(*this)*(*this);
return sqrtl(Rez);
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
    /* Возвращает 0, если массив пуст */
int __fastcall TVectorError::GetLastError(void)
{
if(Count)return iErr[Count-1];
return Count;
}
//---------------------------------------------------------------------------
__fastcall TVectorError::TVectorError()
{
FCount=0;
}
//---------------------------------------------------------------------------
__fastcall TVectorError::~TVectorError()
{
if(Count)delete iErr;
}
//---------------------------------------------------------------------------
    /* Можно получить строку с описанием ошибки, код которой находится в массиве под номером Index */
    /* Index первого элемента = 1 */
AnsiString __fastcall TVectorError::ErrStr(int Index)
{
static char *VectorErrStr[MaxVectorErrCode]={
/* 0*/ "TVectorError::ErrStr() : Вы неправильно указали индекс.",
/* 1*/ "Vector::Create() : Не хватает памяти для массива.",
/* 2*/ "Vector::operator = : Инициализация невозможна, так как размерность вектора не определена.",
/* 3*/ "Vector::operator +(-) : Вектор слагаемое не определен.",
/* 4*/ "Vector::operator +(-) : Несоответствие между размерами векторов."
/* 5*/ "Vector::operator *(/) : Вектор множитель не определен.",
/* 6*/ "Vector::operator * : Несоответствие между размерами векторов множителей.",
/* 7*/ "Унарный Vector::operator - : Вектор не определена.",
/* 8*/ "Vector::operator == : Векторы не определены.",
/* 9*/ "Vector::operator [] : Некорректно заданный индекс"
};
if((Index<=0)||(Index>Count))Index=0;
else Index=iErr[Index-1];
return AnsiString(VectorErrStr[Index]);
}
//---------------------------------------------------------------------------
    /* Очишает массив с ошибками */
void __fastcall TVectorError::Clear(void)
{
if(Count)delete iErr;
FCount=0;
}
//---------------------------------------------------------------------------
    /* Добавить ошибку */
void __fastcall TVectorError::Add(int ErrCode)
{
if((ErrCode<1)||(ErrCode>=MaxVectorErrCode))return;
int *iErrt=new int[Count+1];
int i;
for(i=0;i<Count;i++)iErrt[i]=iErr[i];
if(Count)delete iErr;
iErr=iErrt;
iErr[Count]=ErrCode;
FCount++;
}
//---------------------------------------------------------------------------

