# Simulación de líquidos basada en partículas

Proyecto que permite simular un fluido con partículas usando CUDA y su interoperabilidad con OpenGL. 

## Compilación

Es necesario CMake para compilar el proyecto. Desde la carpeta raíz, se deben ejecutar los siguientes comandos:

- ```cmake -S . -B build``` (solo si no existe la carpeta build)
- ```cmake --build build -j 10``` (cada vez que se quiera compilar el código)

_Disclaimer: el proyecto solo fue probado en Windows, por lo que no se puede garantizar su funcionamiento en otros sistemas operativos._

## Ejecución

Después de compilar el código, se debería crear un ejecutable en la ruta ```build\src\Debug``` llamado _fluid_sim.exe_. Se debe ejecutar desde una línea de comandos, y solo recibe un argumento, el número de partículas. Este número debe ser una potencia de 2 mayor o igual que 64. En caso de no entregar el argumento, se usará el valor por defecto, que son 8192 partículas.

## Controles

Los controles de la simulación son los siguientes:

- Detener/continuar: Espacio
- Menú de opciones: Esc
- Atraer partículas: Click izquierdo
- Alejar partículas: Click derecho

Desde dentro de la simulación, se pueden ajustar los parámetros del fluido mediante el menú de opciones.

## Problemas conocidos

Lamentablemente la simulación no está libre de bugs. Los más importantes son los siguientes:

- Cambiar los parámetros muy rápido puede causar comportamientos erráticos.
- Algunas veces después de ajustar múltiples parámetros pueden generarse 2 zonas en la parte inferior de la ventana que hacen desaparecer las partículas. Si aparecen, la única solución es reiniciar la simulación.
