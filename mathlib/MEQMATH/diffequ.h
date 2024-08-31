//---------------------------------------------------------------------------
#ifndef diffequH
#define diffequH
//---------------------------------------------------------------------------
/* ”казатель на функцию вида y'=f(x,y) */
typedef Extended __fastcall (*HDiffEqu)(Extended&, Extended&);
/* –ешение задачи  оши методом Ёйлера */
Extended __fastcall SolEiler(const HDiffEqu f,const Extended& Xo,const Extended& Yo,const Extended& X,const Cardinal& m,Extended& Eps);
//---------------------------------------------------------------------------
#endif
