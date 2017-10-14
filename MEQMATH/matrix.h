/*
    Описание объекта типа "Матрица"
    08.01.2000
    (c) Хозин Алексей Ю. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#ifndef matrixH
#define matrixH

//---------------------------------------------------------------------------
class Matrix
{
private:
    int FCol;
    int FLin;
    Extended *M; // Массив чисел матрицы
    void __fastcall Free(void); // Высвобождение памяти, Col и Lin не изменяются
    void __fastcall TryCreate(void); // Если Col и Lin не равны нулю, создает массив (Col*Lin) чисел
    void __fastcall SetCol(const int Val);
    void __fastcall SetLin(const int Val);
public:
    /* Конструкторы и деструктор */
    __fastcall Matrix();  // Конструктор по умолчанию
    __fastcall Matrix(const int L, const int C); // Конструктор с созданием таблицы
    __fastcall Matrix(const Matrix& MSrc); // Конструктор создания копии
    __fastcall ~Matrix();
    /* Общие свойства */
    __property int Col = {read = FCol, write = SetCol}; // Колонки
    __property int Lin = {read = FLin, write = SetLin}; // Строки
    /* -= Операторы присваивания =- */
            /* Создание копии (предыдущая информация в матрице теряется) */
    Matrix& __fastcall operator = (const Matrix& MSrc);
            /* Инициализация матрицы массивом чисел (если Src==NULL высвобождение занимаемой памяти, и Lin=Col=0) */
    Matrix& __fastcall operator = (const Extended *Src); // Необходимо зарание задать размеры
    Matrix& __fastcall operator = (const Extended Initial); // Необходимо зарание задать размеры
    /* -= Операторы сложения и вычитания матриц =- */
            /* Размеры матриц должны совпадать */
    Matrix& __fastcall operator += (const Matrix& MSum);
    Matrix __fastcall operator + (const Matrix& MS) const;
    Matrix& __fastcall operator -= (const Matrix& MSum);
    Matrix __fastcall operator - (const Matrix& MS) const;
    /* -= Умножение и деление матриц на число =- */
    Matrix& __fastcall operator *= (const Extended& Val);
    Matrix& __fastcall operator /= (const Extended& Val);
    friend Matrix __fastcall operator * (const Extended& Val, const Matrix& MMul);
    Matrix __fastcall operator * (const Extended& Val) const;
    Matrix __fastcall operator / (const Extended& Val) const;
    /* -= Умножение матриц =- */
            /* Размерности матриц : M1(l,m)*M2(m,n)=M3(l,n) */
    Matrix& __fastcall operator *= (const Matrix& MMul);
    Matrix __fastcall operator * (const Matrix& MMul) const;
    /* Унарные операторы */
    Matrix __fastcall operator - (void) const;
    /* Операторы сравнения */
        /* Равенство только, если : Резмеры совпадают и равны между собой соответствующие элементы */
    bool __fastcall operator == (const Matrix& MCom) const;
    bool __fastcall operator != (const Matrix& MCom) const;
        /* Равенство только, если : Все элементы равны заданному числу */
    bool __fastcall operator == (const Extended& Val) const;
    bool __fastcall operator != (const Extended& Val) const;
    /* Индексный оператор */
    Extended& __fastcall operator [] (int Count); // Count = 1..(Lin*Col)
    /* Фукция для определения индкса элемента по номеру строки и столбца */
    int __fastcall Index(int L, int C);
};
//---------------------------------------------------------------------------
class TMatrixError
{
private:
    int FCount; // Число ошибок
    int *iErr; // Массив кодов ошибок
    int __fastcall GetLastError(void);
public:
    __property int Count = { read = FCount };
    __property int LastErr = { read = GetLastError }; // Возвращает 0, если массив пуст
    __fastcall TMatrixError();
    __fastcall ~TMatrixError();
        /* Можно получить строку с описанием ошибки, код которой находится в массиве под номером Index */
    AnsiString __fastcall ErrStr(int Index); // Index первого элемента = 1
        /* Очишает массив с ошибками */
    void __fastcall Clear(void);
        /* Добавить ошибку */
    void __fastcall Add(int ErrCode);
};
//---------------------------------------------------------------------------
extern TMatrixError *MatrixError;
//---------------------------------------------------------------------------
#endif
