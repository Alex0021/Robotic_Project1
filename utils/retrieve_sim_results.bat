@echo off
SETLOCAL EnableDelayedExpansion

rem Define variables
set "server=alex@alex-ubuntu.homealex.arpa"
set "folder1=/home/alex/Documents/EPFL/Robotic_Project1/sim_results"
set "folder2=/home/alex/Documents/EPFL/Robotic_Project1/exported_plots"
set "local_dir=./sim_results"

rem Check if arguments are provided
if "%~1"=="" (
    rem Retrieve from all folders
    echo "Retrieving files from all folders..."
    scp -r %server%:%folder1% %local_dir%
    echo "Retrieved files from %folder1%"
    scp -r %server%:%folder2% %local_dir%
    echo "Retrieved files from %folder2%"
) else (
    if "%1"=="--data" (
        if "%2"=="" (
            echo "Retrieving data simulation files..."
            rem Retrieve from specific folder
            scp -r %server%:%folder1% %local_dir%
            echo "Retrieved files from %folder1%"
        ) else (
            echo "Retrieving data simulation files from '%2'..."
            rem Retrieve from specific folder
            scp -r %server%:%folder1%/%2 %local_dir%
            echo "Retrieved files from %folder1%/%2"
        )
    ) else (
        if "%1"=="--plots" ( 
            echo "Retrieving plot files..."
            rem Retrieve from specific folder
            scp -r %server%:%folder2% %local_dir%
            echo "Retrieved files from %folder2%"
        ) else (
            echo "Invalid argument. Please use --data or --plots"
            exit /b
        )
    )
)




