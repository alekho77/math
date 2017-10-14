//---------------------------------------------------------------------------
#ifndef integralH
#define integralH
//---------------------------------------------------------------------------
/* ����� ��������������:
    miRectangle - ����� ���������������
    miTrapezoid - ����� ��������
    miSimpson   - ����� ��������
*/
enum TIntegralMethod{miRectangle,miTrapezoid,miSimpson};
//---------------------------------------------------------------------------
/* ��� ��� ����������� ��������������� ������� */
typedef Extended __fastcall (*HUnderIntFunc)(Extended);
//---------------------------------------------------------------------------
/* ��������� ��� �������� ���������� � ������������� ������� */
struct TIntPar
{
    HUnderIntFunc F; // ��������������� �������
    Extended A,B; // ������� ��������������
    Extended Abs; // ���������� �����������
    Extended Eps; // ������������� �����������
    Extended RealEps; // �������� ��������, ���������� ����� ����������
    TIntegralMethod IntMeth; // ����� ��������������
    int iErr; // ������ ������
    __fastcall TIntPar();
    AnsiString __fastcall GetErrStr();
};
//---------------------------------------------------------------------------
Extended __fastcall Integral(TIntPar *IntPar);
Extended __fastcall IntMethRect(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real);
Extended __fastcall IntMethTrap(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real);
Extended __fastcall IntMethSimp(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real);
//---------------------------------------------------------------------------
#endif
