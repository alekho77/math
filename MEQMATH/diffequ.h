//---------------------------------------------------------------------------
#ifndef diffequH
#define diffequH
//---------------------------------------------------------------------------
/* ��������� �� ������� ���� y'=f(x,y) */
typedef Extended __fastcall (*HDiffEqu)(Extended&, Extended&);
/* ������� ������ ���� ������� ������ */
Extended __fastcall SolEiler(const HDiffEqu f,const Extended& Xo,const Extended& Yo,const Extended& X,const Cardinal& m,Extended& Eps);
//---------------------------------------------------------------------------
#endif
