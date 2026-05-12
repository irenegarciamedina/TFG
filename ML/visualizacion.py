import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


try:
    from ML.config import (
        PLOT_RF, REPORT_FILE, OUTPUT_DIR,
        HORIZON_MIN, HORIZON_STEPS, TRAIN_RATIO,
        RF_N_ESTIMATORS,
    )
except ModuleNotFoundError:
    # Ejecutado como script: añadir la raíz del proyecto al path
    _RAIZ_CONFIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _RAIZ_CONFIG not in sys.path:
        sys.path.insert(0, _RAIZ_CONFIG)
    from ML.config import (
        PLOT_RF, REPORT_FILE, OUTPUT_DIR,
        HORIZON_MIN, HORIZON_STEPS, TRAIN_RATIO,
        RF_N_ESTIMATORS,
    )

# mismos colores que la visualización del preprocessing
C1 = "#2980B9"
C2 = "#E74C3C"
C3 = "#27AE60"


# ===========================================================================
# 1. GENERACIÓN DE GRÁFICA Y REPORTE (llamadas desde random_forest.py)
# ===========================================================================

def generar_dashboard_rf(metricas: dict, df: pd.DataFrame) -> None:

    # 3 gráficas
    #ranking de importancia por permutación
    # serie temporal vs lo predicho
    # dispersión real vs lo predicho

    importancias_perm = metricas["importancias_perm"]
    perm_std          = metricas["perm_std"]
    y_test            = metricas["y_test"]
    y_pred_test       = metricas["y_pred_test"]
    rmse_test         = metricas["rmse_test"]
    mae_test          = metricas["mae_test"]
    r2_test           = metricas["r2_test"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # IMPORTANCIA POR PERMUTACIÓN

    ax1    = fig.add_subplot(gs[0, :])
    orden  = importancias_perm.index.tolist()
    vals   = [importancias_perm[f] for f in orden]
    errs   = [perm_std[f] for f in orden]
    cols   = [C1 if v == max(vals) else ("#E67E22" if v > np.mean(vals) else "#BDC3C7") for v in vals]
    bars   = ax1.barh(orden[::-1], vals[::-1], xerr=errs[::-1],
                      color=cols[::-1], capsize=4, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Importancia (reducción de RMSE por permutación)")
    ax1.set_title(
        f"Ranking de importancia de variables — Random Forest\n"
        f"Predicción glucosa a {HORIZON_MIN} min · Paciente HUPA0001P",
        fontweight="bold",
    )
    ax1.axvline(0, color="gray", lw=0.8, ls="--")
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals[::-1]):
        ax1.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)


    # SERIE TEMPORAL VS PREDICHO

    ax2     = fig.add_subplot(gs[1, 0])
    n_plot  = min(len(y_test), 288 * 2)        # máximo 48 h
    inicio  = int(len(df) * TRAIN_RATIO) + HORIZON_STEPS
    idx_test = df.index[inicio : inicio + n_plot]

    ax2.plot(idx_test, y_test[:n_plot],      color=C3, lw=1.0, alpha=0.8, label="Real")
    ax2.plot(idx_test, y_pred_test[:n_plot], color=C2, lw=1.0, alpha=0.8,
             ls="--", label=f"RF ({HORIZON_MIN} min)")
    ax2.axhspan(70, 180, alpha=0.06, color=C3, label="TiR")
    ax2.axhline(70,  color="#E67E22", ls=":", lw=0.8)
    ax2.axhline(180, color="#E67E22", ls=":", lw=0.8)
    ax2.set_title(
        f"Glucosa real vs. predicha (set de test)\n"
        f"RMSE = {rmse_test:.1f} mg/dL | MAE = {mae_test:.1f} mg/dL",
        fontweight="bold", fontsize=9,
    )
    ax2.set_ylabel("Glucosa (mg/dL)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # DISPERSIÓN REAL VS PREDICHO

    ax3  = fig.add_subplot(gs[1, 1])
    ax3.scatter(y_test, y_pred_test, alpha=0.15, s=8, color=C1)
    lims = [min(y_test.min(), y_pred_test.min()) - 5,
            max(y_test.max(), y_pred_test.max()) + 5]
    ax3.plot(lims, lims, "k--", lw=0.8, label="Predicción perfecta")
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.set_xlabel("Glucosa real (mg/dL)")
    ax3.set_ylabel("Glucosa predicha (mg/dL)")
    ax3.set_title(f"Dispersión — R² = {r2_test:.3f}", fontweight="bold", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    plt.suptitle(
        "Fase 3 — Random Forest · Importancia de Features\n"
        "TFG: Predicción de glucosa con LSTM — Irene García Medina",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.savefig(PLOT_RF, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[RF] Gráfica guardada: {PLOT_RF}")


# escritura del reporte

def escribir_reporte_rf(metricas: dict) -> None:
    """Genera el TXT de métricas del Random Forest."""
    imp = metricas["importancias_perm"]

    lineas = [
        "=" * 80,
        "FASE 3 — RANDOM FOREST: RANKING DE IMPORTANCIA DE FEATURES",
        "=" * 80,
        f"Horizonte de predicción : {HORIZON_MIN} minutos ({HORIZON_STEPS} pasos × 5 min)",
        f"Árboles                 : {RF_N_ESTIMATORS}",
        f"División temporal       : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test",
        "",
        "RENDIMIENTO (línea base para comparar con la LSTM):",
        f"  RMSE test  : {metricas['rmse_test']:.2f} mg/dL",
        f"  MAE  test  : {metricas['mae_test']:.2f} mg/dL",
        f"  R²   test  : {metricas['r2_test']:.4f}",
        "",
        "RANKING DE IMPORTANCIA (permutación):",
    ]
    for rank, (feat, val) in enumerate(imp.items(), 1):
        lineas.append(f"  {rank:>2}. {feat:<28} {val:.5f}")
    lineas += [
        "",
        "INTERPRETACIÓN:",
        f"  Feature más influyente   : {imp.index[0]}",
        f"  Feature menos influyente : {imp.index[-1]}",
        "",
        "  -> Las features con importancia < 0 no aportan y pueden excluirse de la LSTM.",
        "=" * 80,
        "",
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mode = "a" if os.path.exists(REPORT_FILE) else "w"
    with open(REPORT_FILE, mode, encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"[RF] Reporte guardado: {REPORT_FILE}")


# ===========================================================================
# 2. VISOR INTERACTIVO TKINTER
# ===========================================================================

# Raíz del proyecto = carpeta padre de ML_Exploratorio/
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _ruta(*partes):
    return os.path.join(RAIZ, *partes)

ARCHIVOS = {
    "Preprocessing":          _ruta("Preprocessing", "output", "Preprocessing.png"),
    "Reporte Preprocessing":  _ruta("Preprocessing", "output", "Preprocessing.txt"),
    "Random Forest":          _ruta("ML_Exploratorio", "output", "RF_importancia_features.png"),
    "Reporte ML":             _ruta("ML_Exploratorio", "output", "ML_Exploratorio_reporte.txt"),
}

BG       = "#1E1E2E"
BG_PANEL = "#2A2A3E"
BG_TAB   = "#252535"
ACCENT   = "#2980B9"
ACCENT2  = "#27AE60"
FG       = "#ECF0F1"
FG_DIM   = "#7F8C8D"
FG_WARN  = "#E67E22"
FG_ERR   = "#E74C3C"

FONT_NORMAL = ("Courier New", 10)
FONT_SMALL  = ("Helvetica", 9)


def _estado_archivos():
    return {nombre: os.path.exists(r) for nombre, r in ARCHIVOS.items()}


def _cargar_imagen(ruta_, max_w, max_h):
    from PIL import Image, ImageTk
    img = Image.open(ruta_)
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class _PestañaImagen:
    """Mixin — construye el contenido de una pestaña PNG."""

    def _placeholder(self, parent_frame, nombre):
        import tkinter as tk
        marco = tk.Frame(parent_frame, bg=BG_PANEL)
        marco.pack(expand=True)
        tk.Label(marco, text="⏳", font=("Helvetica", 48), bg=BG_PANEL, fg=FG_DIM).pack(pady=(40, 8))
        tk.Label(
            marco,
            text=f"'{nombre}' aún no se ha generado.\n\nEjecuta desde la raíz:\n  python main.py",
            font=FONT_SMALL, bg=BG_PANEL, fg=FG_DIM, justify="center",
        ).pack()

    def _build_image(self, parent_frame, nombre, ruta_):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        barra = tk.Frame(parent_frame, bg=BG, pady=6, padx=10)
        barra.pack(fill="x")
        tk.Label(barra, text=os.path.relpath(ruta_, RAIZ),
                 font=FONT_SMALL, bg=BG, fg=FG_DIM).pack(side="left")

        def abrir_completa():
            import tkinter as tk
            from PIL import Image, ImageTk
            v = tk.Toplevel()
            v.title(nombre)
            v.configure(bg=BG)
            img_full = Image.open(ruta_)
            foto = ImageTk.PhotoImage(img_full)
            c = tk.Canvas(v, bg=BG, highlightthickness=0,
                          width=min(img_full.width, 1400),
                          height=min(img_full.height, 900))
            sv = ttk.Scrollbar(v, orient="vertical",   command=c.yview)
            sh = ttk.Scrollbar(v, orient="horizontal", command=c.xview)
            c.configure(yscrollcommand=sv.set, xscrollcommand=sh.set)
            sv.pack(side="right", fill="y"); sh.pack(side="bottom", fill="x")
            c.pack(fill="both", expand=True)
            c.create_image(0, 0, anchor="nw", image=foto)
            c.configure(scrollregion=c.bbox("all"))
            c._img_ref = foto

        tk.Button(barra, text="🔍  Ver a tamaño completo",
                  font=FONT_SMALL, bg=ACCENT, fg="white",
                  relief="flat", padx=10, cursor="hand2",
                  command=abrir_completa).pack(side="right")

        cont = tk.Frame(parent_frame, bg=BG)
        cont.pack(fill="both", expand=True)
        canvas = tk.Canvas(cont, bg=BG, highlightthickness=0)
        sv = ttk.Scrollbar(cont, orient="vertical",   command=canvas.yview)
        sh = ttk.Scrollbar(cont, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=sv.set, xscrollcommand=sh.set)
        sv.pack(side="right", fill="y"); sh.pack(side="bottom", fill="x")
        canvas.pack(fill="both", expand=True)
        canvas.update_idletasks()
        w = max(canvas.winfo_width(),  900)
        h = max(canvas.winfo_height(), 600)
        img_ref = _cargar_imagen(ruta_, w - 20, h - 20)
        canvas.create_image(10, 10, anchor="nw", image=img_ref)
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas._img_ref = img_ref
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        canvas.bind("<Shift-MouseWheel>",
                    lambda e: canvas.xview_scroll(-1 * (e.delta // 120), "units"))
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll( 1, "units"))


def _build_txt_tab(parent_frame, nombre, ruta_):
    import tkinter as tk
    from tkinter import ttk

    if not os.path.exists(ruta_):
        marco = tk.Frame(parent_frame, bg=BG_PANEL)
        marco.pack(expand=True)
        tk.Label(marco, text="📄", font=("Helvetica", 48), bg=BG_PANEL, fg=FG_DIM).pack(pady=(40, 8))
        tk.Label(marco,
                 text=f"'{nombre}' aún no se ha generado.\n\nEjecuta:\n  python main.py",
                 font=FONT_SMALL, bg=BG_PANEL, fg=FG_DIM, justify="center").pack()
        return

    with open(ruta_, encoding="utf-8") as f:
        contenido = f.read()

    barra = tk.Frame(parent_frame, bg=BG, pady=6, padx=10)
    barra.pack(fill="x")
    tk.Label(barra, text="Buscar:", font=FONT_SMALL, bg=BG, fg=FG_DIM).pack(side="left")
    var_buscar = tk.StringVar()
    entrada = tk.Entry(barra, textvariable=var_buscar, font=FONT_SMALL,
                       bg=BG_PANEL, fg=FG, insertbackground=FG, relief="flat", width=28)
    entrada.pack(side="left", padx=(4, 8))
    lbl_res = tk.Label(barra, text="", font=FONT_SMALL, bg=BG, fg=FG_DIM)
    lbl_res.pack(side="right", padx=8)
    tk.Button(barra, text="📋 Copiar todo", font=FONT_SMALL, bg=ACCENT2, fg="white",
              relief="flat", padx=10, cursor="hand2",
              command=lambda: (parent_frame.clipboard_clear(),
                               parent_frame.clipboard_append(contenido),
                               lbl_res.config(text="✓ Copiado", fg=ACCENT2))).pack(side="right")

    marco_txt = tk.Frame(parent_frame, bg=BG)
    marco_txt.pack(fill="both", expand=True)
    sv = ttk.Scrollbar(marco_txt, orient="vertical")
    sh = ttk.Scrollbar(marco_txt, orient="horizontal")
    txt = tk.Text(marco_txt, font=FONT_NORMAL, bg=BG_PANEL, fg=FG,
                  insertbackground=FG, selectbackground=ACCENT,
                  wrap="none", relief="flat", padx=16, pady=12,
                  yscrollcommand=sv.set, xscrollcommand=sh.set)
    sv.configure(command=txt.yview)
    sh.configure(command=txt.xview)
    sv.pack(side="right", fill="y"); sh.pack(side="bottom", fill="x")
    txt.pack(fill="both", expand=True)

    # Highlight
    txt.tag_configure("sep",    foreground=ACCENT,   font=("Courier New", 10, "bold"))
    txt.tag_configure("titulo", foreground=FG,        font=("Courier New", 10, "bold"))
    txt.tag_configure("clave",  foreground=ACCENT2)
    txt.tag_configure("valor",  foreground="#F39C12")
    txt.tag_configure("flecha", foreground=FG_WARN)
    txt.tag_configure("normal", foreground=FG)
    for linea in contenido.splitlines(keepends=True):
        s = linea.strip()
        if set(s) <= {"=", "-", " "} and len(s) > 4:
            txt.insert("end", linea, "sep")
        elif s.isupper() and len(s) > 3:
            txt.insert("end", linea, "titulo")
        elif s.startswith("→") or s.startswith("->"):
            txt.insert("end", linea, "flecha")
        elif ":" in linea and not linea.startswith(" " * 4):
            partes = linea.split(":", 1)
            txt.insert("end", partes[0] + ":", "clave")
            txt.insert("end", partes[1] if len(partes) > 1 else "", "valor")
        else:
            txt.insert("end", linea, "normal")
    txt.configure(state="disabled")

    def buscar():
        termino = var_buscar.get().strip()
        txt.tag_remove("busqueda", "1.0", "end")
        if not termino:
            lbl_res.config(text="")
            return
        txt.tag_configure("busqueda", background="#F39C12", foreground="#1E1E2E")
        start, count, primera = "1.0", 0, None
        while True:
            pos = txt.search(termino, start, stopindex="end", nocase=True)
            if not pos:
                break
            end = f"{pos}+{len(termino)}c"
            txt.tag_add("busqueda", pos, end)
            if primera is None:
                primera = pos
            start = end
            count += 1
        if primera:
            txt.see(primera)
        lbl_res.config(text=f"{count} resultado{'s' if count != 1 else ''}",
                       fg=ACCENT2 if count > 0 else FG_ERR)

    tk.Button(barra, text="🔍", font=FONT_SMALL, bg=ACCENT, fg="white",
              relief="flat", padx=8, cursor="hand2",
              command=buscar).pack(side="left")
    entrada.bind("<Return>", lambda _: buscar())


def _run_visor():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Visor TFG — Predicción de glucosa · HUPA0001P")
    root.geometry("1280x820")
    root.minsize(900, 600)
    root.configure(bg=BG)

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure("TNotebook",     background=BG,     borderwidth=0)
    s.configure("TNotebook.Tab", background=BG_TAB, foreground=FG_DIM,
                padding=[14, 6], font=("Helvetica", 10))
    s.map("TNotebook.Tab",
          background=[("selected", BG_PANEL)],
          foreground=[("selected", FG)])
    s.configure("TFrame",                  background=BG)
    s.configure("Vertical.TScrollbar",   background=BG_PANEL, troughcolor=BG)
    s.configure("Horizontal.TScrollbar", background=BG_PANEL, troughcolor=BG)

    cab = tk.Frame(root, bg=BG, pady=10, padx=16)
    cab.pack(fill="x")
    tk.Label(cab, text="Resultados del TFG",
             font=("Helvetica", 16, "bold"), bg=BG, fg=FG).pack(side="left")
    tk.Label(cab, text="Predicción de glucosa con LSTM · HUPA0001P",
             font=FONT_SMALL, bg=BG, fg=FG_DIM).pack(side="left", padx=16)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=(0, 4))

    PESTANAS = [
        ("PREPROCESADO",          None),
        ("Preprocessing",         ARCHIVOS["Preprocessing"]),
        ("Reporte Preprocessing", ARCHIVOS["Reporte Preprocessing"]),
        ("MACHINE LEARNING",          None),
        ("Random Forest",         ARCHIVOS["Random Forest"]),
        ("Reporte ML",            ARCHIVOS["Reporte ML"]),
    ]

    mixin = _PestañaImagen()

    def añadir_pestañas():
        for nombre, ruta_ in PESTANAS:
            if ruta_ is None:
                sep = ttk.Frame(nb)
                nb.add(sep, text=f" {nombre} ", state="disabled")
                continue
            existe = os.path.exists(ruta_)
            frame  = ttk.Frame(nb)
            nb.add(frame, text=f"  {nombre}{'  ⚠' if not existe else '  '}")
            if ruta_.endswith(".txt"):
                _build_txt_tab(frame, nombre, ruta_)
            elif existe:
                mixin._build_image(frame, nombre, ruta_)
            else:
                mixin._placeholder(frame, nombre)

    añadir_pestañas()

    # Barra de estado
    barra = tk.Frame(root, bg=BG, pady=6, padx=12)
    barra.pack(fill="x", side="bottom")

    def dibujar_barra():
        for w in barra.winfo_children():
            w.destroy()
        tk.Label(barra, text="Archivos:", font=FONT_SMALL, bg=BG, fg=FG_DIM).pack(side="left", padx=(0, 10))
        for nombre, existe in _estado_archivos().items():
            tk.Label(barra,
                     text=("● " if existe else "○ ") + nombre,
                     font=FONT_SMALL, bg=BG,
                     fg=ACCENT2 if existe else FG_ERR).pack(side="left", padx=5)

        def refrescar():
            for tab_id in nb.tabs():
                nb.forget(tab_id)
            añadir_pestañas()
            dibujar_barra()

        tk.Button(barra, text="↺ Refrescar", font=FONT_SMALL, bg=BG_PANEL, fg=FG_DIM,
                  relief="flat", padx=8, cursor="hand2",
                  command=refrescar).pack(side="right")

    dibujar_barra()
    root.mainloop()


if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("\n[Visor] ERROR: Pillow no está instalado.")
        print("  Instálalo con:  pip install pillow\n")
        sys.exit(1)

    _run_visor()