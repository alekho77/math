//---------------------------------------------------------------------------
#include <vcl.h>
#include <math.h>
#pragma hdrstop

#include "diffequ.h"
//---------------------------------------------------------------------------
Extended __fastcall SolEiler(const HDiffEqu f,const Extended& Xo,const Extended& Yo,const Extended& X,const Cardinal& cm,Extended& Eps)
{
Cardinal i,m=cm;
Extended Xi,h,Yi,Y;
if(m<1)m=1; // На случай ошибки
Eps=fabsl(Eps);
/* Начальный расчет */
h=(X-Xo)/m;
Xi=Xo;
Yi=Yo;
for(i=0;i<m;i++)
  {
  Xi=Xi+h;
  Yi=Yi+h*((*f)(Xi,Yi));
  }
/* Пересчитываем с вдвое меньшим шагом */
do{
  Y=Yi;
  m*=2;
  h=(X-Xo)/m;
  Xi=Xo;
  Yi=Yo;
  for(i=0;i<m;i++)
    {
    Xi=Xi+h;
    Yi=Yi+h*((*f)(Xi,Yi));
    }
  }while(Eps<fabsl(Y-Yi));
return Yi;
}
//---------------------------------------------------------------------------
#pragma package(smart_init)
