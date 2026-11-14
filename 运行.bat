@echo off
chcp 936 >nul
cd /d "%~dp0"
echo ==================================================
echo            气温小实验
echo   三个问题，不想改就直接按回车
echo ==================================================
echo.
set CITY=
set /p CITY=城市（1成都 2北京 3上海 4广州 5武汉 6西安）:
set STY=
set /p STY=起始年份（例如 2016）:
set LAM=
set /p LAM=约束强度（默认 1，可试 0.1 或 10）:
set TGT=
set /p TGT=预测最高温还是最低温（回车=最高，输 min =最低）:

set ARGS=
if not "%CITY%"=="" set ARGS=%ARGS% --city %CITY%
if not "%STY%"=="" set ARGS=%ARGS% --start %STY%-01-01
if not "%LAM%"=="" set ARGS=%ARGS% --lam %LAM%
if /i "%TGT%"=="min" set ARGS=%ARGS% --target tmin

echo.
echo 开始跑：python run.py %ARGS%
echo.
python run.py %ARGS%
echo.
echo 结果在 outputs 文件夹，图已经自动打开了。
pause
