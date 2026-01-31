# scripts/patch_mmpose.py
import os

def patch_setup_py():
    target_file = os.path.join("models", "mmpose", "setup.py")
    
    if not os.path.exists(target_file):
        print(f"❌ Error: No se encuentra {target_file}")
        return

    print(f"🔧 Parcheando {target_file} para compatibilidad con PEP 517...")

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # El código antiguo que causa el error
    bad_code = "exec(compile(f.read(), version_file, 'exec'))"
    
    # El código nuevo (robusto)
    good_code = "version_vars = {}; exec(compile(f.read(), version_file, 'exec'), version_vars); locals().update(version_vars)"

    # Verificamos si ya está parcheado
    if good_code in content:
        print("✅ El archivo ya estaba parcheado.")
        return

    # Verificamos si encontramos el código viejo
    if bad_code not in content:
        # A veces el formato cambia ligeramente, buscamos una alternativa común
        if "locals()['__version__']" in content:
             print("⚠️ Advertencia: No se encontró el bloque exacto, intentando reemplazo agresivo...")
             # Reemplazo manual por bloques si fuera necesario, pero para MMPose 1.3.2 esto suele bastar
        else:
            print("❌ Error: No se encuentra el patrón de código a reemplazar. Revisa la versión de MMPose.")
            return

    # Aplicar el reemplazo
    new_content = content.replace(bad_code, good_code)
    
    # Adicionalmente, arreglamos el return para que no busque en locals() genérico si falla
    new_content = new_content.replace("return locals()['__version__']", "return version_vars['__version__']")

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ ¡Parche aplicado con éxito!")

if __name__ == "__main__":
    patch_setup_py()