from tkinter import simpledialog
import numpy as np
import tkinter as tk
from tkinter import ttk  # Import ttk for Notebook (Tabs)
from tkinter import filedialog
from tkinter import messagebox
from tkinter.colorchooser import askcolor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk
import collections # For color history deque

# --- Tooltip Class (Unchanged) ---
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        self.schedule_id = self.widget.after(500, self.create_tooltip)

    def create_tooltip(self):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if hasattr(self, 'schedule_id'):
            self.widget.after_cancel(self.schedule_id)
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# -----------------

class NPYImageEditor:
    def __init__(self, master):
        self.root = master
        self.root.title("NPY Image Editor")

        self.current_mouse_position = tk.StringVar()
        self.current_mouse_position.set("x=--, y=--")

        self.image = None
        self.zoom_level = 1.0
        self.current_color = [0, 0, 0, 255]
        self.selected_rgba_color = self.current_color
        
        # --- NEW: Color History ---
        self.color_history = collections.deque(maxlen=16)
        # --------------------------

        # --- NEW: Grid State ---
        self.show_grid = False
        # -----------------------

        self.mouse_pressed = False
        self.tool = None
        self.selected = None
        self.selection_start = None
        self.selection_end = None
        self.selection_mode = "new"
        self.selection_preset = None
        
        self.state_queue = []
        self.current_state_index = -1
        self.default_canvas_size = (600, 400) 
        self.cmd_pressed = False

        # --- ICONS (Keeping your existing base64 data) ---
        # (I will use placeholders for the new Grid icon to keep code clean, 
        # but you can replace the base64 string)
        
        self.save_icon_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABdElEQVR4nO2Xv07DMBCHP6dJxFAJXoGBlQdhRKyw8RZU/BEvgMQzILFQCUYGdhZeAgnEhMRAW5qGwXfEOLHaoNAu/kmnqy/2+avvaqUAN0AhVorNOrIpMAbOsUoBg6fyn20mftAEkcpDn2oiCxeR8dbrxgA92aMAzmR84sRK40xGgj3gAHgAkgVARt4cA/Ql1y5wgS0FsvGpC5EGkr4Az3M2Vh0Ce9hTy4E74FKevTnzegJyLGOF+FWvqfgd7LfPxDdZLv7ay3Erp5AA+xL7ouoH/TzQI2mSTjaES6Cd/oE97hGwJmMXyJWRk9CeyJNA8jZKJKnavJwKUQJHXQD8VQYouwDQC0etaAMR6gG32fw6+nPWsb3Ul/gGVRlqt56vEMAnVZOFNBF/BbxjuzsD7p1147YASrwFvGKbJQShN+gT8OjEM2AbW45NL29N/k24dK3yVxABIkAEiAARIAJEgB+AVb4P1P6aLV0JMGT++1/X0v2G35Nljn+igX7oAAAAAElFTkSuQmCC"
        self.save_icon = tk.PhotoImage(data=self.save_icon_data)
        
        self.open_icon_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACAklEQVR4nL2Xu0oEMRSGv5kdFbEQK/HyEKK9pa21vU+gD+IrWImVgmDhDUQLK621E0QtBC+FgrI7Y5FznGx2ZjNroj8czu5kkvMn5z9JBpojFYuKBJho+O6L+BbQiUngxfNOLv4QWAMehUQOFDFIFAPYA7Bi9W3JJJpYLfKG1raIbAFTAZP+QcJgy1gImRYmFeuY1ECZqrp+b1WxBiWg6AgJ6K+hQmK8AvPiu2JmnkB15FLKSmhSRaqVHvgI9BNPq09bY/gIfAIfAePbKahczToNaI5XgR0M0XYgkUoRaqNtusG8A5MBQV3U7g8uAa33M+lYK6AYqNKALpPWtyp+GBiLHD+tIqDqPhKvs9/AbMNt/OL1QcV5r3/UOuJvgRGLwAhwR2+6Qm3TnUmOWfJTTAkOA1/AHDArBGPcCXQ733UJ6HIfiNdgS+ILwgVZSPAn4Fwf2OX3AUzLy6qHc7orJMR0jD17hlCeZleYcz+TZzPAgrTFWH67ypK0omHfebYIjGLyH2M/0NvUCVBkTkMBXGJOOBXgskMwBCrya+AGSGwCOrttp9O4+NDatwkcy++satCmt+TfQFN+YD+sOoxsi7Xp6FjPlJNMMvy5jXL1ptzCLzDXuBTIM/7wpHMwJN7e5PIM/4dJDOgq5pSHXA6DfZrFIvL6j/H88H42/QFiiToOvgG4j9n07GmfLQAAAABJRU5ErkJggg=="
        self.open_icon = tk.PhotoImage(data=self.open_icon_data)
        
        self.zoom_in_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAADLElEQVR4nM3XSYicRRQH8N90z+K4EEfJKG4EccklrmgS8SCIC8ZD9KA55SYKnkxILh70Ho+KoEQU9WZydQHHZQK5mEH0kriAkoMHwQQTHRMz0x7qPfrrL193z7RM9MFHdb31X1Xvvaoes3oaq/zujGA/ErUx3sBvBb+1VoFbelcMF2MdJmv8doPuUBpk0MZS/H4Q27EF12MKv+M7fI4DOFoBvbxaIE3BRcA55azzO4c/a7wzeBVX1uxHojzrZ3E2AvyIl7AV12IGN2Ib3goAHWUXbgv7kfIikT+ju7p9ypk3gUzahM9C/xfcMgqIDH63svJlPFeRT4XOXhxXVtrWTcZJHAwQC6HflMR9KZUPhZNXgj8R/AR4IORPx3y8IpvGtyF/oSIfSunggTD+SSm3ammlzntKdTxRC5Djw+Hj5wA0ZsAu5BmlwlNhvF/J9Gwy7drXxEsQn+Br3KAkbceAXEhB1vuWAPNxzM+EbEnJixzhdMxT51x88GEE3lpb4Hk0HsKOsuXXhZPjwd+DO5SEbIfefWH7InaG3nLI3g3wx4K/oV/gKoCkCSWTz2Ix5i8r59hE9zfwWgHgr5inbd9LqwpgEaeURrMOJ7EZN4eDbM27Ivg+HA4fS8ouHFZWPhM+T8Q4tBQzF7LtPjLA6O2KTj96M3Sej3nfUmzVxrkYd4SDSd2reKoywmUxvyjGidC9BI+H/RehO/RySgA3KVn9B24NXrs2vo+/8WRtddkR90TweWUXV9yOM8Br4eBLZVUZJAN9EPIEkMlLaeOnQv5Qze9QSrQz+CGcHMSlFUfjytnvx9V6z/YepXw7eL0CfFWU27VJudU6SlfbNsDmcmXbT+veoPOYrQDvS02ZnuW2Ee/g3uAfwadKk1nEFbgLj+Ka0HkDtyvluxCyX/W+rlZEiXoKu/G93hdQ/ZvDY2FzVQTvBOj1NZ89NKhBVN9200pf36y012n8prwJD+Gb0JtQKmQWH+FO/3Inqu+AlerlOKu7EwtWmBPDAmQ5Vn831fkwEGv2P6IfiCP/BxBf6SbmpCEvpbUAsV65LecvNAB6n34XLGid/rPAVTpvB/4Bb2G5yZFPJYsAAAAASUVORK5CYII="
        self.zoom_in_icon = tk.PhotoImage(data=self.zoom_in_data)
        
        self.zoom_out_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAC30lEQVR4nM3Xy49URRQG8F/3vDJgAkSBhEeiCa8FCGaiwI5AIAQ3/hNGt8Sww8geloCJC4hhLVtlARqGxI0MBBY81ER04cJEjYLAAHNZ1Cn79qX79p3m+SWVvlV1Hl+dOqequmX+aJW+iyH0h8IIRnuMt2O8/bwct3WvGBZgEcYr4yM9ZAeiTmEEj+J7Fz7ANqzGBP7BTXyHr3C9RHpuvkR6ORcOz0l7ndtD3KmM3cdRvF7RHwp5rz/CbDj4GZ9hO1bgDbyF93EiCBRSFN4O/aHyIjP/UGd1h6U9r8MmfBvyv2PdMCSy8ylp5XP4uDQ/GjI5Mdu6q2Mcp4PEjJQnvZK4L7LwhTByJMbHGhjJ5CdxNfT3l4gPRDawI5R/kcptPqWVHe0JG7eCUKuJjax8XAr9p9GfCBJN2mjJzqUgsbOywCeQkyTX+7Zgeyb692OuSXsYDb4OAtuj3zcCozFZSCFfFUZ+i/ED2BJRqcvoXDGngvyN0H+zRud/AhljUibP4m70D0n72BTtIHAv+lm376VVJnAX/2KlVPd/YyvWhoFBiTSH70NuSYz9Fb+1W1AE81n8KG3DBvwqldTVAY57YSp+r9dKVYjAwSB0MvrjujO8ro2F7ELpNJzDxrAz8ETMAmukzL+D9THW9GLJ1/OBWMS0zonZCNnRsTBwXloV9adZq+R8SsqjArsrdgcis12Cn8LIabxWmq+GvWz8Xal8C3zegHhP5HBtkvaxwGXp2u2XyYulsN/WOQ+msSzmayPQy2h+CW3Al3gvxi/iLK7hP+lNMIW90hsBvsBmqXxnYu4P3a+rRsisJ/CJVJ5FTTuHfaGzPJwXQXppxWYX6g6X8ttuUjrXt0rH6yT+lN6EF3Al5MbwQAr/N3jHU0ai1Y95jVz+XaYTiRkNc2KQg3L25+9edT6IxHP7H9GPxMVXgcQPOok5ruFL6VmSWCrdltMvmgCdcL9Qp1W8NMdlPBGBx/VXqCrJTq4vAAAAAElFTkSuQmCC"
        self.zoom_out_icon = tk.PhotoImage(data=self.zoom_out_data)
        
        self.pencil_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACR0lEQVR4nL3XT4iNURjH8c/9NwwzkZQUEsnWguUsZCk7O7HHQv4sLCRkQVJslYWUlfInjST5k2I2SmwmUZKMbBTRjHHH4pzTe+Ztppu5772/Or33Offc8z3Pc859zvPSP9XQyOxG7OsbPGkYy/oFhnp8NnEZE/iGKxjqB7yGFu5gptRu9BoOA7gXgVNoxzaNv72GL8JoBk+etyP8dy/hi3F/DvgM/sTn6V7BB/GwBEstLeZSr+BL8GgeeLITvKEiJfhSPO4Av5jBK0lECT6EJx3gF3oFH8azOeDtzD4Xxzargqf9W4bnHeBnq4anSZbjRQf4marhjdjW42UH+Kmq4fkk+yJkMkLL8JNVw9OBWyH81+FwhKW8Ph3tE1XBaxHcjPYqjAv5vRX7jpm9DcergJerF1iNN4q9viXcdHBEiMDRDL5g5asewGaM4LXC0+TtbUUktsZnXRdK8LrgzbjZGa2dfU6XyqgiEv8Nz0NVU1QvN7Er9o/FVseBOKat2KKnQvhbQmQWrDThecGzd9hRGrNbKCLSFuyP/V2FPU1Qx9oI+IqN2cKaQnUD1yL8YLRbulAzA0xir1DJHMJ7YW+nSov8Eu0H0a6srtuGH/gs3OuNDJq8rOGtkHDWZAvrSntwNcJ/YnvsL+eBlbiuOPmVwAkHK/2lxrAu+24QW4RL5VMcMyHkhpQlK9EGRRE5gw9C1pswOw/cxab4m8re6/JQ7xTeVj7iF77jlfBKNZKNq8zzfMKyR0PCYcxVadjnUio2ymrO09+1/gHVZMGquu7zPAAAAABJRU5ErkJggg=="
        self.pencil_icon = tk.PhotoImage(data=self.pencil_data)

        self.magic_wand_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACnklEQVR4nL3Xv49NURDA8c9bD4uEbTQiIRG/Co2OUinEj4IoaOwPi/9As1EQeiqFCg2FxL+g8CtKiRARBdGJH7FrPcWdyT3vsrx9761Jbs69886d+d6ZOXPOo1uWxVU+w3F8wU20MFLMGYl5LQNKaSAdJMB9/MQnjBXzS5C+JA200MEe7A1nIwXUs7h/ia+F85/YgP26I7coyRf3BkQHk43fluMANjb0O/Em3rkcuvagAHMxniicl9JWRWBT4byDq/0CUKdisjB4R3eOt2BtMf9gMfcuVuhOW98QJ3AD2+J5Ox7iM97iVOhXq8J+RR3Fvpzn0mv7vZB24p36S5s1UkqrsNV3QVLlvI0dDedPMKuq/A6mYv6oAXvAMVW+Z8LYsoB4XDi/H/pJzKsL9UwBPRo27uBw6HvqE2Vo88Wt4WQ+QKiKTDjp4FuM50N/tLDzXVUj9BCdm3iPR9gcL4zho7r7Zbh34YU6Aj9iHMc6PMUHXFdFrOdOOVaQLo/7k7qjcxGvddfEXAMC1vfqNKWkbFZv9oX84mZNnI7nWd01scIiJZcQdW9PI1PqnJc1kR1yogExHfpFdcSMQrO3p/6MOhKfwimsijEj1YxETxDZPhfq7aMxngt99oFsRgND5IS/9fZMTzPnQ4NoYSUuqXp7eU5IA4dUq2V8qSD+BJVfPhPGnsdz1sTQIdrq/V4B8EC9EnKdn10qiFIS4IhqY7qmSk8u0X9FYsqQ0kHd26kilEb+C0SzKFP6heirWS20qy0EkRtYE6K5dwxFeoWYVrXzvHb/T4g1Mc6Efh4XhgnwN4jMeZ4nMg1Hhw3wJ4jcyi/ilbrd3zbg4bUXiGndG1he9wzpD20vEBPq82MHtwrnrSUlCEdZ7ftUR7p78VsLnV+hsfuN/rgq+gAAAABJRU5ErkJggg=="
        self.magic_wand_icon = tk.PhotoImage(data=self.magic_wand_data)

        self.rect_select_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACKElEQVR4nO2WzWoVQRCFv/m5XggkJuJOIch9Cp/BhZCYhwkuxI2+gW7EdZCIK/ExDNm4TcCVihGM4o3JHRd92qnp9PwnKOiBonrOdHVXV1d3F/zHH0ZSw6eSAjgzfC59arhM4ywkTfZ/P1LpGfAM2DT8FHgkmZq+m+o7M+PE7DvBh3gHF75jYEXcXXGF2ujfsbgdM461XxJXt90VZNL3gEPgibgUWAfeStbFZepzKBsi9j4femPZtL33mXHSrmiJ81iOcJ0RC1fS0k5r/g+aOAPuADcCPmlxrs2+FT4JH+OSaB+X8XUTj7YflByXibEhHL0F4UBDMdYeGHmMhtrbi+iA6kXUx34DeA88pedFFLuK+1yl3oFXsv8KrDbZ58G3f07va/I3wHfcChZ0x1z9v9VNPBQ5553OjFyRfoGLwAfgurjc6MqAfXAa4c4i7RPpAjiioSgJHfChngHbuC14KX4CPFC/h8BP9d0AtnCh9lXQbfW7CuwahxLgHe6mnDc51KcemBu+q3j7PIxAIb2rVbzGJVIK7EmQTvXvuaJQUB63FdwbsAA+m4nB1Ql7lHVkLewb31YPrAJruIRbozyGH4Fb4q5JKqeiLglT3PFLjPcJZTJ53re/BPY/pAvgE+4+sPhtX+fAIpjED5aYdozP5KTtN6F8jovQvukYFh05y/toTeXIJPK/gouuB/zK93HROMCdpAu9Dbs4kOPqgZsB/+8gwUWhdYt/AZVmjyIoHY1fAAAAAElFTkSuQmCC"
        self.rect_select_icon = tk.PhotoImage(data=self.rect_select_data)
        
        self.dropper_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACsklEQVR4nM2XS29NURTHf/e2vc1FgmivREhIxGNQIhdtZ0JIg4GhLyBMRQzFB2DokRhIfACdMlGPNjHREgYaj4QYGEhI0Iqix2D9l3Pucd60rORk77POevz32mutvU+N8lSLzIMK+pWoC+hO4NfFry+U4zqdKwZYAiwHGjF+V4JsLmUpdAE/NN8HHAGGgHVAL/AReAbcAa4D0xHQ82WBJDlHDsewvfbnOzAT430FLgCrYvqVyPf6ODAnBy+Bs8AwsAboAzYAh4CrAhBgUdgm/Up54ciPEa7uHLbnWTQA3Jb8W2BTFRDuvI2tfB44EfneLRlPzDqd1dEARgViCsuTpCROJReekJHz4vcUMOLgm8AT6Z+MAM8lN7BHyq+wcitTWu7ogGy8FqBaERuufAkL/ZkYvyj5nj8UiL16T60KV/B6H8LQ3tRYttW6vRvSHdZ7agR83wMs5GuBb8Ab8ao2lKeyu74oYrBka2AVMFfRsdOsxqbG1EhGAXwBPmGRWCZe6d4unT7NP+TZqQtdHVv1cwlvJaz1MuTNa6fepzNkfwGIjmMaj1I+AT2XlgKHNb+rb7m55AA2Yn19BtgsXtGDxY/n03I+TskouqOLMnAPS0zI7ge1iPM2lkcBsD9mN5cc7UrghYyM0pmQ3bEnanwXYfleLgA8kTxcA9ipFgCPsGM3LZNXYGH/TJiE40BL3zMjkGTUb0JbgGvAbvEngVtYk5nFSq0NjGB3BIArwHZgEDsRR4B3dN6uCpGj7gVOYeUZZDxjwEHprJbzQKD7YzY7KKvRRO92TayvD2LttQm8x+6EE8BjyfVgrbyFnSc7+MNI1NKQZ8j52CKMxBQFcyLPQTT7fZ5U53kgFuw/Ig3E5P8A4gFhYjYoeFP6myD6gftYj1hUABCGe1GdxumfOY7SbxH4CTJflNcFG2UUAAAAAElFTkSuQmCC"
        self.dropper_icon = tk.PhotoImage(data=self.dropper_data)

        self.zoom_reset_data = self.dropper_data 
        self.zoom_reset_icon = tk.PhotoImage(data=self.zoom_reset_data)

        self.select_color_data = self.magic_wand_data
        self.select_color_icon = tk.PhotoImage(data=self.select_color_data)

        self.vflip_icon_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAEq0lEQVR4nM1XX2ibVRT/nXvvlyYtpQVZtUuWzLDm22JF1KoPwqI4RJt2g0H2sBdxMBB8mNMHUR9ihQk+yNSn+aK+iEhe/NPO4RwjwvYwu8FQ4tqVQGq7yXDaWtOm+b57jw/Jl6ZZE+ycrQe+P1zOPb/fvffc8wdYJWmB/1xWY1DDP9+zazAijYoycYc0kE3tkGEIi8gxV6envspVDI8Y297buUTYzYaVp6oFNDEVtXDzv/58suBhAYBaYTViQnbyFYI6JpTVBiLQKn6NwiCScPTCYm/v0Jbr11ECgCL0m7627peNWwaoMl+AAWaQoeWQnXxjZmLsXQ+TPDbR6J6usvLPgqgd4JNgLjGR0xyfWUhFbNzxmYmB40COgIwO7tr7mGDxojGu8lZAzBaI/AANgnnRb9ytU1On/kSVIgHgUCgV4PbFnJC+oKvdJ69Pfn2uxfLXLb2x4ceVVGeNKc+o0l/xQiFbQm2PUimJTEYHY8mXlNV+XLtLeeHjgWndXkQgz4hGTVPLGQDI6JWBtEAqR8gAeDgvsBSlsFzsMGUalyoQdZ3Fo7OTY+95mN4sAtICiYQKxZLnI/0pDsYGPwIAJBIKtyvVucHY4CcVm8nzlbG0QPUCeMYZACGbdWVs6JB2Sxekans+1Dd4eiZ78jPseKYNU50uEjdaeeWKZHsYOxYUsqeWQ/bQQSGt57RbWlCgQ8hmXeAJUcVswrgveTgc38/b7OHfIva+7be7AeHY8L3b7OGb4fh+DvYlD9djeHLrihIJhWzWDdrJzy2r44DjFH8g4H0Glchwc1+oExYkCOxn4IhldTziOMXM7MTYAc92awIVf6Dw/eNduiy+t6xAP8Br7VdLoerbcZZ+kj6ze/rHgXlghNGw9c3OlADw1p1P3SW54wUGxxjsWx8BKhNoUpM4ce3KFzdRF/1aka577mReqHl9/VMD9b5rs/OcJtvDSAG4cYMaz3GVbk8PIwPUbkwz3SpmjUkkkvCju9uPuTqVbqBw+cu5+gkAEI3u6dKdHVTT7QbkQpHz+e/mG3UjD+zrbrSJublSNRJWdiC8c+ghA/EN2PjBVVJEhkgKw+6lLtn+dC4HDWR00E6+LoT1KmvHALVsqUlawhjnndmJsbeBlIzHIefdxdNCqAeZtQGzqNplIlEimGenr4xe2oD831r+H0eATXTCxoFNu4ZrstuIQLTpofjWXJ9ISGRHXF1OnrCsQP8dSUbl4ofAyJrJqBH8jqbjiL1v+zrScaVKjcSGdmpBF4RQncZZPjhz9V8UJFO1guRTY9wFafjRwuToFQ+rgcjmlGT/rCi9GDVIAcjEeTXzlESqBYH8GkWpu3RkdmL0Aw9zQ8tyS6qz2ji/qNLCfYVCdhmouwU+3x++Mvxb2LhKEr8WiiVXGhMGCyGNNvrja5OjZ4CUBOIcssePklADRrsMoqb+UWlM2G+Mq4hxt2UFfABKAEgBYCAt8vmR+ZCdTJOQx4SwkqtaM2YI5QOX5gyAM0Cce3sv+sF4S6lAO0uN5jGtGkWYoY2zDNdNV7qildasRhStmlMShuXSuZnct797k8M79sbZEn0wDoNFUwatmtMG2fj2/G+NVp8I4AoNGQAAAABJRU5ErkJggg=="
        self.vflip_icon = tk.PhotoImage(data=self.vflip_icon_data)

        self.hflip_icon_data = "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"
        self.hflip_icon = tk.PhotoImage(data=self.hflip_icon_data)

        self.rotate_icon_data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACkklEQVR4nMXXS6hNURgH8N859yhvCYmBPEZKKXkkA0lhxoiBxFQmBoYMpDs0VMYMpEQxQGGsDCgGYuL9Si55pe7DYK1197r7nnPP3se5+ddq77X2933/b33fenyb/4xGjzrNTHesZGc0tr6jiYF+y1aNwABG4vt8bMEGrIr9P3iLx7iPl5kjY7H1hBRuWImzeJUZbde+4Rp2ZHaaekBDEaHjGGpD9hOf8bWDMxexKNqomr5x8iZauFAy+hyD2ClEZTGWYxOO4V5J/hnW1nGikQleiUZG8QMnMKeCjd14kjnxDqvjt67pSOSDUXkEr7Exk2lFubQd84glgnnCWkhOPMJME7dwR/KNUWkYXxQhnDGVcslBUfZW5sRgiWcSkve3M6UDGXkdJJLFeC9E8hdWlLgmka+PwqO4G8daZeGKSHpHFRM608lmGjidCe8VwtirA2l9zMcHYVKPTNzi40gRuBPJP2KBEMqBdgoVkVJxSXF2jKchkTaid63s4zPhVBuJrdfjNM32QezPxpr0rbwQ5gjbBz4JUbmAm8Jh0zZ0FTAmLMSEhWUH2oX5G7bhEPZgXzRU60jthlYkHY79H/gd35fgiCIF9+N4L3d9A8uy/lB8jqd1s3C2N3E9fhhSXDIPM0N1MeUihO2KbfcYT7N+WnynokLd7dh1G5YNrovPYcW2HMHV+F43/APR1kEsjWM3hEm1FKm3FeeEq7Z8p0/wuCY5FY7i3PAs7MJ5vBBKrcMlg1VQ+zJqmpyOuULB0Q19vY5TMdIqjXVCp+LinwqSnLiK8Fx9LsmqIBnaL1TJX0xevP9UlHZDStHlNqS1yvJe7/l0HpzEd2F2b4S8t/sxmZZftanQ91+zqYjK4Z22n9NpwV9RzcSsJey9ggAAAABJRU5ErkJggg=="
        self.rotate90_icon = tk.PhotoImage(data=self.rotate_icon_data)
        # ------------------------------------

        self.create_widgets()
        
        # Bind keyboard shortcuts
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-a>", self.select_all)
        self.root.bind("<Control-d>", self.deselect_all)
        self.root.bind("<Control-s>", self.save_full_image) 
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-MouseWheel>", self.on_mouse_scroll)
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<Key>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

    def on_key_press(self, event):
        if event.keysym in ('Control_L', 'Control_R', 'Control'):
            self.cmd_pressed = True

    def on_key_release(self, event):
        if event.keysym in ('Control_L', 'Control_R', 'Control'):
            self.cmd_pressed = False

    def create_widgets(self):
        # Menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Selection", command=lambda: self.save_selection(None))
        file_menu.add_command(label="Save full image", command=lambda: self.save_full_image(None), accelerator="Ctrl+S")
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")

        edit_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_menu, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo_menu, accelerator="Ctrl+Y")

        select_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Select", menu=select_menu)
        select_menu.add_command(label="Rectangular Selection", command=self.activate_rectangular_selection)
        select_menu.add_command(label="Magic Wand", command=self.activate_magic_wand)
        select_menu.add_command(label="Select w/ Same Color", command=self.select_all_with_color)
        select_menu.add_separator()
        select_menu.add_command(label="Select All", command=lambda: self.select_all(None), accelerator="Ctrl+A")
        select_menu.add_command(label="Deselect All", command=lambda: self.deselect_all(None), accelerator="Ctrl+D")

        # --- NEW: Canvas Resize Menu ---
        canvas_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Canvas", menu=canvas_menu)
        canvas_menu.add_command(label="Add Row (Top)", command=lambda: self.resize_canvas('top', 'add'))
        canvas_menu.add_command(label="Add Row (Bottom)", command=lambda: self.resize_canvas('bottom', 'add'))
        canvas_menu.add_command(label="Add Column (Left)", command=lambda: self.resize_canvas('left', 'add'))
        canvas_menu.add_command(label="Add Column (Right)", command=lambda: self.resize_canvas('right', 'add'))
        canvas_menu.add_separator()
        canvas_menu.add_command(label="Remove Row (Top)", command=lambda: self.resize_canvas('top', 'remove'))
        canvas_menu.add_command(label="Remove Row (Bottom)", command=lambda: self.resize_canvas('bottom', 'remove'))
        canvas_menu.add_command(label="Remove Column (Left)", command=lambda: self.resize_canvas('left', 'remove'))
        canvas_menu.add_command(label="Remove Column (Right)", command=lambda: self.resize_canvas('right', 'remove'))
        # -------------------------------
        
        main_toolbar_frame = tk.Frame(self.root)
        main_toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        # File buttons
        open_btn = tk.Button(main_toolbar_frame, image=self.open_icon, command=self.open_file)
        open_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(open_btn, "Open File (Ctrl+O)")

        save_btn = tk.Button(main_toolbar_frame, image=self.save_icon, command=lambda: self.save_full_image(None))
        save_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(save_btn, "Save File (Ctrl+S)")

        tk.Frame(main_toolbar_frame, width=2, height=24, bg="grey").pack(side=tk.LEFT, padx=5)

        # Zoom buttons
        zoom_in_btn = tk.Button(main_toolbar_frame, image=self.zoom_in_icon, command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(zoom_in_btn, "Zoom In")
        
        zoom_out_btn = tk.Button(main_toolbar_frame, image=self.zoom_out_icon, command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(zoom_out_btn, "Zoom Out")

        zoom_reset_btn = tk.Button(main_toolbar_frame, image=self.zoom_reset_icon, command=self.reset_zoom)
        zoom_reset_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(zoom_reset_btn, "Reset Zoom (1:1)")

        tk.Frame(main_toolbar_frame, width=2, height=24, bg="grey").pack(side=tk.LEFT, padx=5)
        
        # Tools
        pencil_btn = tk.Button(main_toolbar_frame, image=self.pencil_icon, command=self.activate_pencil)
        pencil_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(pencil_btn, "Pencil")
        
        dropper_btn = tk.Button(main_toolbar_frame, image=self.dropper_icon, command=self.activate_dropper)
        dropper_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(dropper_btn, "Color Dropper")

        # --- NEW: Toggle Grid Button ---
        self.grid_btn = tk.Button(main_toolbar_frame, text="Grid", command=self.toggle_grid)
        self.grid_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(self.grid_btn, "Toggle Pixel Grid")
        # -------------------------------

        tk.Frame(main_toolbar_frame, width=2, height=24, bg="grey").pack(side=tk.LEFT, padx=5)

        rect_select_btn = tk.Button(main_toolbar_frame, image=self.rect_select_icon, command=self.activate_rectangular_selection)
        rect_select_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(rect_select_btn, "Rectangular Selection")

        magic_wand_btn = tk.Button(main_toolbar_frame, image=self.magic_wand_icon, command=self.activate_magic_wand)
        magic_wand_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(magic_wand_btn, "Magic Wand")

        select_color_btn = tk.Button(main_toolbar_frame, image=self.select_color_icon, command=self.select_all_with_color)
        select_color_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(select_color_btn, "Select All w/ Same Color")

        vflip_btn = tk.Button(main_toolbar_frame, image=self.vflip_icon, command=self.vertical_flip)
        vflip_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(vflip_btn, "Vertical Flip")
        
        hflip_btn = tk.Button(main_toolbar_frame, image=self.hflip_icon, command=self.horizontal_flip)
        hflip_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(hflip_btn, "Horizontal Flip")

        rotate90_btn = tk.Button(main_toolbar_frame, image=self.rotate90_icon, command=self.rotate_90)
        rotate90_btn.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(rotate90_btn, "Rotate 90° CCW")

        # Color Tools Frame
        tools_frame = tk.Frame(self.root)
        tools_frame.pack(side=tk.TOP, fill=tk.X) 

        self.color_indicator = tk.Label(tools_frame, text="", bg=self.rgb_to_hex(self.current_color[:3]),
                                        width=5, relief=tk.RAISED)
        self.color_indicator.pack(side=tk.LEFT, padx=5)
        self.color_indicator.bind("<Button-1>", self.open_color_palette)
        Tooltip(self.color_indicator, "Click to change color")

        self.alpha_indicator = tk.Label(tools_frame, text=f"a: {self.current_color[3]}", width=5, relief=tk.RAISED)
        self.alpha_indicator.pack(side=tk.LEFT, padx=5)
        self.alpha_indicator.bind("<Button-1>", self.open_alpha_input)
        Tooltip(self.alpha_indicator, "Click to change alpha (opacity)")

        # --- NEW: Transparent Color Button ---
        transparent_btn = tk.Button(tools_frame, text="Transparent", command=self.set_transparent_color)
        transparent_btn.pack(side=tk.LEFT, padx=5)
        Tooltip(transparent_btn, "Set current color to fully transparent")
        # -------------------------------------

        self.tool = None

        # Selection Mode Frame
        self.selection_mode_frame = tk.Frame(self.root)
        self.selection_mode_frame.pack(side=tk.TOP, fill=tk.X) 
        self.selection_mode_frame.pack_forget()

        tk.Label(self.selection_mode_frame, text="Selection Mode:").pack(side=tk.LEFT, padx=5)
        tk.Button(self.selection_mode_frame, text="New", command=lambda: self.set_selection_mode("new")).pack(side=tk.LEFT)
        tk.Button(self.selection_mode_frame, text="Add", command=lambda: self.set_selection_mode("add")).pack(side=tk.LEFT)
        tk.Button(self.selection_mode_frame, text="Subtract", command=lambda: self.set_selection_mode("subtract")).pack(side=tk.LEFT)
        tk.Button(self.selection_mode_frame, text="Intersect", command=lambda: self.set_selection_mode("intersect")).pack(side=tk.LEFT)
        tk.Button(self.selection_mode_frame, text="Fill Selected", command=lambda: self.fill_selected()).pack(side=tk.LEFT)
        tk.Button(self.selection_mode_frame, text="Save as Preset", command=lambda: self.save_selection_preset()).pack(side=tk.LEFT)
        tk.Button(self.selection_mode_frame, text="Load from Preset", command=lambda: self.load_selection_preset()).pack(side=tk.LEFT)

        status_bar = tk.Label(self.root, textvariable=self.current_mouse_position, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # --- MODIFIED: Right Sidebar with Tabs (History / Colors) ---
        self.sidebar_frame = tk.Frame(self.root, width=220)
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        self.sidebar_tabs = ttk.Notebook(self.sidebar_frame)
        self.sidebar_tabs.pack(expand=True, fill=tk.BOTH)

        # Tab 1: Undo/Redo History
        self.history_tab = tk.Frame(self.sidebar_tabs)
        self.sidebar_tabs.add(self.history_tab, text="History")
        
        self.state_canvas = tk.Canvas(self.history_tab, width=200, height=300)
        self.state_canvas.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(self.history_tab, orient=tk.VERTICAL, command=self.state_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.state_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.image_thumbnails = []

        # Tab 2: Colors (Palette & History)
        self.colors_tab = tk.Frame(self.sidebar_tabs)
        self.sidebar_tabs.add(self.colors_tab, text="Colors")
        
        # Used Colors (History)
        tk.Label(self.colors_tab, text="Recent Colors:").pack(anchor="w", padx=5, pady=2)
        self.color_history_frame = tk.Frame(self.colors_tab)
        self.color_history_frame.pack(fill=tk.X, padx=5)
        
        # Unique Image Colors
        tk.Label(self.colors_tab, text="Image Colors:").pack(anchor="w", padx=5, pady=(10, 2))
        
        # Scrollable frame for image colors
        self.palette_canvas = tk.Canvas(self.colors_tab)
        self.palette_scrollbar = tk.Scrollbar(self.colors_tab, orient=tk.VERTICAL, command=self.palette_canvas.yview)
        self.palette_scroll_frame = tk.Frame(self.palette_canvas)
        
        self.palette_scroll_frame.bind(
            "<Configure>",
            lambda e: self.palette_canvas.configure(scrollregion=self.palette_canvas.bbox("all"))
        )
        
        self.palette_canvas.create_window((0, 0), window=self.palette_scroll_frame, anchor="nw")
        self.palette_canvas.configure(yscrollcommand=self.palette_scrollbar.set)
        
        self.palette_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.palette_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # -------------------------------------------------------------

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(self.canvas_frame)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")

        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.v_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.scroll_canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self.scroll_canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_window = self.scroll_canvas.create_window(0, 0, anchor=tk.NW, window=self.canvas_widget)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)

        self.scroll_canvas.bind("<Configure>", self.update_scroll_region)

    # --- NEW: Set Transparent Color ---
    def set_transparent_color(self):
        self.set_current_color([0, 0, 0, 0])

    # --- NEW: Helper to set color and update UI/History ---
    def set_current_color(self, rgba):
        self.current_color = list(rgba)
        
        # Update Color History (only if not already the most recent)
        if not self.color_history or list(self.color_history[-1]) != self.current_color:
            self.color_history.append(tuple(self.current_color))
            self.update_palette_ui()

        # Update Indicators
        if self.current_color[3] == 0:
            # Indicate transparency
            self.color_indicator.config(bg="white", text="Transp")
        else:
            self.color_indicator.config(bg=self.rgb_to_hex(self.current_color[:3]), text="")
            
        self.alpha_indicator.config(text=f"a: {self.current_color[3]}")

    # --- NEW: Toggle Grid ---
    def toggle_grid(self):
        self.show_grid = not self.show_grid
        self.grid_btn.config(relief=tk.SUNKEN if self.show_grid else tk.RAISED)
        self.update_display()

    # --- NEW: Update Palette and History UI ---
    def update_palette_ui(self):
        # Clear existing widgets in palettes
        for widget in self.color_history_frame.winfo_children():
            widget.destroy()
        for widget in self.palette_scroll_frame.winfo_children():
            widget.destroy()

        # 1. Render Recent History (Right to Left = Newest to Oldest)
        # We reverse to show newest on the left/top
        for i, color_tuple in enumerate(reversed(self.color_history)):
            if i > 15: break # double check limit
            c = list(color_tuple)
            self.create_color_button(self.color_history_frame, c).pack(side=tk.LEFT, padx=1, pady=1)

        # 2. Render Image Unique Colors
        if self.image is not None:
            # Reshape image to list of pixels, find unique rows
            pixels = self.image.reshape(-1, 4)
            unique_colors = np.unique(pixels, axis=0)
            
            row, col = 0, 0
            max_cols = 4
            for color_arr in unique_colors:
                c = list(color_arr)
                btn = self.create_color_button(self.palette_scroll_frame, c)
                btn.grid(row=row, column=col, padx=1, pady=1)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

    def create_color_button(self, parent, color):
        # Check if transparent
        if color[3] == 0:
            bg_color = "white"
            text = "T"
        else:
            bg_color = self.rgb_to_hex(color[:3])
            text = ""
            
        btn = tk.Button(parent, bg=bg_color, text=text, width=2, relief=tk.FLAT,
                        command=lambda c=color: self.set_current_color(c))
        if color[3] == 0:
            Tooltip(btn, "Transparent")
        else:
            Tooltip(btn, f"R:{color[0]} G:{color[1]} B:{color[2]} A:{color[3]}")
        return btn

    # --- NEW: Canvas Resize Logic ---
    def resize_canvas(self, direction, action):
        if self.image is None: return
        
        h, w = self.image.shape[:2]
        new_image = self.image
        new_selected = self.selected if self.selected is not None else np.zeros((h, w), dtype=bool)

        # Define an empty row/col (transparent)
        # Note: We fill with transparent, or current BG color? Standard is transparent.
        
        try:
            if direction == 'top':
                if action == 'add':
                    empty_row = np.zeros((1, w, 4), dtype=np.uint8)
                    new_image = np.vstack([empty_row, self.image])
                    new_selected = np.vstack([np.zeros((1, w), dtype=bool), new_selected])
                elif action == 'remove':
                    if h <= 1: raise ValueError("Cannot remove last row")
                    new_image = self.image[1:, :, :]
                    new_selected = new_selected[1:, :]
            
            elif direction == 'bottom':
                if action == 'add':
                    empty_row = np.zeros((1, w, 4), dtype=np.uint8)
                    new_image = np.vstack([self.image, empty_row])
                    new_selected = np.vstack([new_selected, np.zeros((1, w), dtype=bool)])
                elif action == 'remove':
                    if h <= 1: raise ValueError("Cannot remove last row")
                    new_image = self.image[:-1, :, :]
                    new_selected = new_selected[:-1, :]
            
            elif direction == 'left':
                if action == 'add':
                    empty_col = np.zeros((h, 1, 4), dtype=np.uint8)
                    new_image = np.hstack([empty_col, self.image])
                    new_selected = np.hstack([np.zeros((h, 1), dtype=bool), new_selected])
                elif action == 'remove':
                    if w <= 1: raise ValueError("Cannot remove last column")
                    new_image = self.image[:, 1:, :]
                    new_selected = new_selected[:, 1:]

            elif direction == 'right':
                if action == 'add':
                    empty_col = np.zeros((h, 1, 4), dtype=np.uint8)
                    new_image = np.hstack([self.image, empty_col])
                    new_selected = np.hstack([new_selected, np.zeros((h, 1), dtype=bool)])
                elif action == 'remove':
                    if w <= 1: raise ValueError("Cannot remove last column")
                    new_image = self.image[:, :-1, :]
                    new_selected = new_selected[:, :-1]

            self.image = new_image
            self.selected = new_selected
            self.update_state(f"Resize {direction} {action}")

        except ValueError as e:
            messagebox.showwarning("Resize Error", str(e))

    # -----------------------------------------------------------

    def set_selection_mode(self, mode):
        self.selection_mode = mode

    def fill_selected(self):
        if self.selected is not None and self.image is not None:
            self.image[self.selected] = self.current_color
            self.update_display()
            self.update_state("fill_selected")

    def update_state(self, last_step_name=None):
        new_state = self.get_current_state(last_step_name)
        if self.current_state_index < len(self.state_queue) - 1:
            self.state_queue = self.state_queue[0 : self.current_state_index + 1]
        self.state_queue.append(new_state)
        self.current_state_index = len(self.state_queue) - 1
        self.update_display()
        self.update_state_queue_display()
        
        # --- NEW: Update Palette when state changes (e.g. image loaded, color painted)
        self.update_palette_ui()

    def update_state_queue_display(self):
        self.state_canvas.delete("all")
        self.image_thumbnails = []
        y_position = 10
        for idx, state in enumerate(self.state_queue):
            image_thumb = self.create_thumbnail(state["image"])
            self.image_thumbnails.append(image_thumb)
            self.state_canvas.create_image(10, y_position, anchor=tk.NW, image=image_thumb)
            text_color = "grey" if idx > self.current_state_index else "black"
            self.state_canvas.create_text(60, y_position + 10, anchor=tk.NW, text=state["last_step_name"], fill=text_color)
            y_position += 60
        self.state_canvas.config(scrollregion=self.state_canvas.bbox("all"))
        self.state_canvas.yview_moveto(1)

    def get_current_state(self, last_step_name=None):
        return {
            "image": self.image.copy(),
            "zoom_level": self.zoom_level,
            "current_color": self.current_color.copy(),
            "selected": self.selected.copy() if self.selected is not None else None,
            "last_step_name": self.tool if last_step_name is None else last_step_name,
        }

    def load_state(self, state):
        self.image = state["image"].copy()
        self.zoom_level = state["zoom_level"]
        self.current_color = state["current_color"].copy()
        self.selected = state["selected"].copy() if state["selected"] is not None else None
        self.update_display()

    def undo(self, _):
        if self.current_state_index > 0:
            self.current_state_index -= 1
            state_to_load = self.state_queue[self.current_state_index]
            self.load_state(state_to_load)
            self.update_state_queue_display()
        else:
            print("No more steps to undo.")

    def undo_menu(self):
        self.undo(None)

    def redo(self, _):
        if self.current_state_index < len(self.state_queue) - 1:
            self.current_state_index += 1
            state_to_load = self.state_queue[self.current_state_index]
            self.load_state(state_to_load)
            self.update_state_queue_display()
        else:
            print("No more steps to redo.")

    def redo_menu(self):
        self.redo(None)

    def create_thumbnail(self, image):
        img_pil = Image.fromarray(image)
        img_resized = img_pil.resize((50, 50))
        return ImageTk.PhotoImage(img_resized)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if not file_path:
            return
        try:
            self.root.title(f"NPY Image Editor - {file_path}")
            self.image = np.load(file_path)
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                print("RGB format detected. Converts to RGBA format.")
                self.image = np.dstack([self.image, np.ones(self.image.shape[:2], dtype=np.uint8) * 255])
            elif len(self.image.shape) != 3 or self.image.shape[2] != 4:
                raise ValueError("Invalid NPY file format. Expected 3D array with 4 channels (RGBA).")

            original_height, original_width = self.image.shape[:2]
            canvas_width, canvas_height = self.default_canvas_size
            width_ratio = canvas_width / original_width
            height_ratio = canvas_height / original_height
            self.zoom_level = min(width_ratio, height_ratio)

            self.selected = np.zeros(self.image.shape[:2], dtype=bool)
            self.update_state("open file") # This calls update_display and update_palette_ui
        except (ValueError, IOError) as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_selection(self, _):
        y, x = np.where(self.selected)
        if len(y) == 0:
            messagebox.showwarning("Warning", "No pixel is selected. Please select a region to be saved.")
            return
        y0, y1 = np.min(y), np.max(y)
        x0, x1 = np.min(x), np.max(x)
        region_to_save = self.image[y0 : y1 + 1, x0 : x1 + 1].copy()
        selected_pixels = self.selected[y0 : y1 + 1, x0 : x1 + 1]
        region_to_save[~selected_pixels] = [0, 0, 0, 0]

        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
        if file_path:
            np.save(file_path, region_to_save)
            messagebox.showinfo("Saved", f"Selection saved to {file_path}")
        else:
            messagebox.showwarning("Warning", "Invalid Path.")

    def save_full_image(self, _):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
        if file_path:
            np.save(file_path, self.image)
            messagebox.showinfo("Saved", f"Image saved to {file_path}")
        else:
            messagebox.showwarning("Warning", "Invalid Path.")

    def select_all(self, _):
        if self.image is not None:
            self.selected = np.ones(self.image.shape[:2], dtype=bool)
            self.update_state("select all")

    def deselect_all(self, _):
        if self.image is not None:
            self.selected = np.zeros(self.image.shape[:2], dtype=bool)
            self.update_state("deselect")

    def zoom_in(self):
        self.zoom_level *= 1.5
        self.update_display()

    def zoom_out(self):
        self.zoom_level /= 1.5
        self.update_display()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.update_display()

    def activate_pencil(self):
        self.tool = "pencil"
        self.selection_mode_frame.pack_forget()

    def activate_magic_wand(self):
        self.tool = "magic_wand"
        self.selection_mode_frame.pack(fill=tk.X)

    def activate_rectangular_selection(self):
        self.tool = "rectangular_selection"
        self.selection_mode_frame.pack(fill=tk.X)

    def activate_dropper(self):
        self.tool = "dropper"
        self.selection_mode_frame.pack_forget()

    def select_all_with_color(self):
        self.tool = "select_all_with_color"
        self.selection_mode_frame.pack(fill=tk.X)

    def vertical_flip(self):
        if self.image is None: return
        self.image = np.flipud(self.image)
        if self.selected is not None:
            self.selected = np.flipud(self.selected)
        self.update_state("Vertical Flip")

    def horizontal_flip(self):
        if self.image is None: return
        self.image = np.fliplr(self.image)
        if self.selected is not None:
            self.selected = np.fliplr(self.selected)
        self.update_state("Horizontal Flip")

    def rotate_90(self):
        if self.image is None: return
        self.image = np.rot90(self.image, k=1, axes=(0, 1))
        if self.selected is not None:
            self.selected = np.rot90(self.selected, k=1, axes=(0, 1))
        self.update_state("Rotate 90° CCW")
    
    def save_selection_preset(self):
        if self.selected is not None:
            self.selection_preset = self.selected.copy()

    def load_selection_preset(self):
        if self.selection_preset is not None:
            self.selected = self.selection_preset.copy()

    def open_color_palette(self, event=None):
        color = askcolor(color=self.rgb_to_hex(self.current_color[:3]))[0]
        if color:
            r, g, b = color
            a = self.current_color[3]
            self.set_current_color([int(r), int(g), int(b), a])

    def open_alpha_input(self, event=None):
        alpha = simpledialog.askinteger("Input", "Enter an integer between 0 and 255", parent=self.root, minvalue=0, maxvalue=255)
        if alpha is not None:
            self.current_color[3] = alpha
            self.set_current_color(self.current_color)

    def on_mouse_press(self, event):
        if self.image is None or event.xdata is None or event.ydata is None:
            return

        self.mouse_pressed = True
        x, y = int(event.xdata), int(event.ydata)

        if self.tool == "pencil":
            self.image[y, x] = self.current_color
            self.update_display()
        elif self.tool == "rectangular_selection":
            self.selection_start = (y, x)
        elif self.tool == "dropper":
            color = self.image[y, x].tolist()
            self.set_current_color(color)

    def on_mouse_release(self, event):
        if self.mouse_pressed and event.xdata and event.ydata:
            if self.tool == "rectangular_selection":
                if self.selection_start is None:
                    self.mouse_pressed = False
                    return
                self.selection_end = (int(event.ydata), int(event.xdata))
                y0 = min(self.selection_start[0], self.selection_end[0])
                x0 = min(self.selection_start[1], self.selection_end[1])
                y1 = max(self.selection_start[0], self.selection_end[0])
                x1 = max(self.selection_start[1], self.selection_end[1])
                selected_pixels = np.zeros(self.image.shape[:2], dtype=bool)
                selected_pixels[y0 : y1 + 1, x0 : x1 + 1] = True
                self.submit_selection(selected_pixels)
                self.update_state("rectangular_selection")

            if self.tool == "pencil":
                self.update_state("pencil")
            if self.tool == "magic_wand":
                new_selection = self.magic_wand(int(event.ydata), int(event.xdata), self.image[int(event.ydata), int(event.xdata)])
                if self.cmd_pressed:
                    if self.selected is None:
                        self.selected = new_selection
                    else:
                        self.selected = np.logical_or(self.selected, new_selection)
                else:
                    self.selected = new_selection
                self.update_state("magic_wand")

            if self.tool == "select_all_with_color":
                new_selection = np.all(self.image == self.image[int(event.ydata), int(event.xdata)], axis=2)
                self.submit_selection(new_selection)
                self.update_state("select with same color")

        self.selection_start = None
        self.selection_end = None
        self.mouse_pressed = False

    def delete_selected(self, _):
        if self.selected is not None:
            self.image[self.selected] = [0, 0, 0, 0]
            self.update_display()
            self.update_state("delete selected")

    def on_mouse_scroll(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def magic_wand(self, y, x, target_color):
        stack = [(y, x)]
        new_selection = np.zeros(self.image.shape[:2], dtype=bool)
        if not (0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]):
            return new_selection
        new_selection[y, x] = True
        processed = {(y, x)}
        while stack:
            cur_y, cur_x = stack.pop()
            for ny, nx in [(cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)]:
                if (not (0 <= ny < self.image.shape[0] and 0 <= nx < self.image.shape[1]) or (ny, nx) in processed):
                    continue
                processed.add((ny, nx))
                if np.all(self.image[ny, nx] == target_color):
                    new_selection[ny, nx] = True
                    stack.append((ny, nx))
        return new_selection

    def on_mouse_motion(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.current_mouse_position.set(f"x={x}, y={y}")
            if self.tool == "pencil" and self.mouse_pressed:
                if self.selection_start:
                    prev_y, prev_x = self.selection_start
                    num_steps = max(abs(x - prev_x), abs(y - prev_y))
                    if num_steps == 0: num_steps = 1
                    for i in range(num_steps + 1):
                        inter_x = int(prev_x + (x - prev_x) * i / num_steps)
                        inter_y = int(prev_y + (y - prev_y) * i / num_steps)
                        if (0 <= inter_y < self.image.shape[0] and 0 <= inter_x < self.image.shape[1]):
                            self.image[inter_y, inter_x] = self.current_color
                self.selection_start = (y, x)

            if self.tool == "rectangular_selection" and self.mouse_pressed:
                self.selection_end = (int(event.ydata), int(event.xdata))
        else:
            self.current_mouse_position.set("x=--, y=--")

        self.update_display()

    def update_canvas_size(self):
        if self.image is not None:
            original_height, original_width = self.image.shape[:2]
            new_width = int(original_width * self.zoom_level)
            new_height = int(original_height * self.zoom_level)
            self.canvas_widget.config(width=new_width, height=new_height)
            self.scroll_canvas.itemconfig(self.canvas_window, width=new_width, height=new_height)
            self.scroll_canvas.config(scrollregion=(0, 0, new_width, new_height))

    def update_scroll_region(self, event=None):
        self.scroll_canvas.config(scrollregion=self.scroll_canvas.bbox("all"))

    def update_display(self):
        if self.image is None:
            return

        self.ax.clear()
        
        # --- MODIFIED: Grid Drawing Logic ---
        if self.show_grid:
            # Draw ticks between pixels, then hide labels
            height, width = self.image.shape[:2]
            # Major ticks at 0.5, 1.5, etc.
            self.ax.set_xticks(np.arange(0.5, width, 1), minor=True)
            self.ax.set_yticks(np.arange(0.5, height, 1), minor=True)
            self.ax.grid(which='minor', color='white', linestyle='-', linewidth=1, alpha=0.5)
            self.ax.tick_params(which='minor', size=0) # Hide tick markers
            
            # Hide the major ticks/numbers (axes) so only the image and grid remain
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            # Also remove axis splines if you want a cleaner look
            self.ax.axis("on") 
        else:
            self.ax.axis("off")
        # ------------------------------------

        self.update_canvas_size()

        self.ax.imshow(self.image)

        if self.selection_start and self.selection_end and self.tool == "rectangular_selection":
            y0 = min(self.selection_start[0], self.selection_end[0])
            x0 = min(self.selection_start[1], self.selection_end[1])
            y1 = max(self.selection_start[0], self.selection_end[0])
            x1 = max(self.selection_start[1], self.selection_end[1])
            rect = Rectangle((x0 - 0.5, y0 - 0.5), (x1 - x0) + 1, (y1 - y0) + 1, linewidth=1, edgecolor="r", facecolor="none", linestyle='--')
            self.ax.add_patch(rect)

        if self.selected is not None and self.selected.any():
            overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
            overlay[self.selected] = [0, 0, 255, 128]
            self.ax.imshow(overlay, interpolation="none")

        self.canvas.draw()

    def rgba_to_hex(self, rgba):
        rgba = np.array(rgba) / 255
        rgb = (1 - rgba[3]) + rgba[3] * rgba[:3]
        rgb = (rgb * 255).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def rgb_to_hex(self, rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def submit_selection(self, selection):
        if self.selected is None:
             self.selected = np.zeros(self.image.shape[:2], dtype=bool)
        if self.selection_mode == "new":
            self.selected = selection
        elif self.selection_mode == "add":
            self.selected = np.logical_or(self.selected, selection)
        elif self.selection_mode == "subtract":
            self.selected = np.logical_and(self.selected, np.logical_not(selection))
        elif self.selection_mode == "intersect":
            self.selected = np.logical_and(self.selected, selection)

if __name__ == "__main__":
    root = tk.Tk()
    app = NPYImageEditor(root)
    root.mainloop()