import sys
import os
import traceback

# Prevent problematic xformers, peft, and torchaudio conflicts with local DLLs or ComfyUI on Windows.
sys.modules['xformers'] = None
sys.modules['peft'] = None
sys.modules['torchaudio'] = None

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, 
                             QTextEdit, QCheckBox, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont

class ConverterThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, input_path, output_path, use_half, use_safetensors):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.use_half = use_half
        self.use_safetensors = use_safetensors

    def run(self):
        try:
            self.log.emit("🔄 Loading core libraries (PyTorch & Diffusers)...")
            self.log.emit("This might take a moment if libraries are cached or compiled.")
            
            try:
                import torch
                import transformers
                from diffusers import StableDiffusionXLPipeline
                
                # Monkeypatch transformers to avoid 'CLIPTextModel' has no attribute 'text_model'
                # during from_single_file initialization
                def get_text_model(self):
                    if '_modules' in self.__dict__ and 'text_model' in self.__dict__['_modules']:
                        return self.__dict__['_modules']['text_model']
                    for name, child in self.named_children():
                        if name == 'text_model':
                            return child
                        if hasattr(child, 'embeddings'):
                            return child
                    if hasattr(self, 'embeddings'):
                        return self
                    return self

                if hasattr(transformers, 'CLIPTextModel'):
                    transformers.CLIPTextModel.text_model = property(get_text_model)
                if hasattr(transformers, 'CLIPTextModelWithProjection'):
                    transformers.CLIPTextModelWithProjection.text_model = property(get_text_model)
                    
            except Exception as e:
                self.log.emit("\n❌ LIBRARY IMPORT ERROR DETECTED!")
                self.log.emit(f"Detail: {str(e)}")
                self.log.emit(traceback.format_exc())
                
                self.log.emit("\n💡 [TROUBLESHOOTING GUIDE]")
                self.log.emit("==================================================")
                self.log.emit("There was an issue loading PyTorch or Diffusers.")
                self.log.emit("We have automatically bypassed both 'xformers' and 'peft' internally")
                self.log.emit("to completely prevent conflicts, leaving your ComfyUI setup safe and untouched!")
                self.log.emit("If you still see errors, try updating packages inside PowerShell with ComfyUI closed:")
                self.log.emit("\n   pip install --upgrade diffusers torch accelerate transformers PyQt6")
                self.log.emit("==================================================")
                
                self.finished.emit(False, "Import error occurred. See logs for solution.")
                return

            self.log.emit(f"Starting conversion for: {os.path.basename(self.input_path)}")
            self.log.emit("Loading StableDiffusionXLPipeline from single file...")
            self.log.emit("NOTE: This may take a few minutes and consume significant RAM.")
            
            self.progress.emit(10)
            
            dtype = torch.float16 if self.use_half else torch.float32
            self.log.emit(f"Using dtype: {dtype}")
            self.progress.emit(20)
            
            # Load pipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                self.input_path, 
                torch_dtype=dtype, 
                use_safetensors=True,
                low_cpu_mem_usage=False
            )
            
            # Clean up the monkeypatch to prevent any serialization or save_pretrained conflicts
            if hasattr(transformers.CLIPTextModel, 'text_model'):
                try:
                    del transformers.CLIPTextModel.text_model
                except Exception:
                    pass
            if hasattr(transformers.CLIPTextModelWithProjection, 'text_model'):
                try:
                    del transformers.CLIPTextModelWithProjection.text_model
                except Exception:
                    pass

            self.progress.emit(50)
            self.log.emit("Inspecting pipeline components for any meta/empty tensors...")
            
            # Repair function to find and materialize any parameters/buffers left on the 'meta' device
            def fix_meta_tensors(module):
                import torch.nn as nn
                for param_name, param in list(module.named_parameters(recurse=False)):
                    if param.device.type == "meta":
                        self.log.emit(f"  Materializing unassigned param: {param_name}")
                        new_param = nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cpu"))
                        setattr(module, param_name, new_param)
                for buffer_name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.device.type == "meta":
                        self.log.emit(f"  Materializing unassigned buffer: {buffer_name}")
                        new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                        setattr(module, buffer_name, new_buf)
                for child_name, child in module.named_children():
                    fix_meta_tensors(child)

            for component_name, component in pipe.components.items():
                if component is not None and hasattr(component, "named_parameters"):
                    fix_meta_tensors(component)

            self.progress.emit(70)
            self.log.emit("Model loaded and prepared in memory.")
            self.log.emit(f"Saving unrolled diffusers format to: {self.output_path}")
            
            # Save pipeline
            pipe.save_pretrained(
                self.output_path, 
                safe_serialization=self.use_safetensors
            )
            
            self.progress.emit(100)
            self.log.emit("✅ Conversion completed successfully!")
            self.finished.emit(True, "Success")
            
        except Exception as e:
            # Check for OOM specifically
            err_str = str(e)
            if "OutOfMemoryError" in err_str or "CUDA out of memory" in err_str:
                self.log.emit("\n❌ ERROR: Out of GPU Memory (VRAM) / CPU RAM.")
                self.log.emit("Try closing other applications or ensure 'Half Precision (fp16)' is checked.")
                self.finished.emit(False, "Out of Memory")
            else:
                self.log.emit(f"\n❌ ERROR: {err_str}")
                self.log.emit(traceback.format_exc())
                self.finished.emit(False, err_str)


class SDXLConverterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SDXL Safetensors to Diffusers Converter')
        self.resize(650, 450)
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
                background-color: #f7f9fa;
                color: #1a1a1a;
            }
            QPushButton {
                background-color: #0f172a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #334155; }
            QPushButton:disabled { background-color: #94a3b8; }
            QLabel { font-weight: 500; }
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
            }
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_label = QLabel("SDXL to Diffusers Converter")
        header_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # Input selection
        in_layout = QHBoxLayout()
        self.in_btn = QPushButton("Browse Safetensors...")
        self.in_btn.setFixedWidth(160)
        self.in_btn.clicked.connect(self.select_input)
        self.in_path_label = QLabel("No file selected")
        self.in_path_label.setStyleSheet("color: #64748b; font-style: italic;")
        in_layout.addWidget(self.in_btn)
        in_layout.addWidget(self.in_path_label, 1)
        layout.addLayout(in_layout)
        
        # Output selection
        out_layout = QHBoxLayout()
        self.out_btn = QPushButton("Set Output Folder...")
        self.out_btn.setFixedWidth(160)
        self.out_btn.clicked.connect(self.select_output)
        self.out_path_label = QLabel("No directory selected")
        self.out_path_label.setStyleSheet("color: #64748b; font-style: italic;")
        out_layout.addWidget(self.out_btn)
        out_layout.addWidget(self.out_path_label, 1)
        layout.addLayout(out_layout)
        
        # Options
        options_layout = QHBoxLayout()
        self.half_check = QCheckBox("Use Half Precision (fp16) - Saves RAM")
        self.half_check.setChecked(True)
        options_layout.addWidget(self.half_check)
        
        self.safetensors_check = QCheckBox("Output as Safetensors format")
        self.safetensors_check.setChecked(True)
        options_layout.addWidget(self.safetensors_check)
        layout.addLayout(options_layout)
        
        # Convert button
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.setStyleSheet("background-color: #2563eb; font-size: 11pt;")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)
        layout.addWidget(self.convert_btn)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        
        self.setLayout(layout)
        
        self.input_file = None
        self.output_dir = None

    def select_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select SDXL Safetensors File", 
            "", 
            "Safetensors (*.safetensors);;All Files (*)"
        )
        if file_path:
            self.input_file = file_path
            self.in_path_label.setText(os.path.basename(file_path))
            self.in_path_label.setStyleSheet("color: #0f172a;")
            self.check_ready()

    def select_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.out_path_label.setText(dir_path)
            self.out_path_label.setStyleSheet("color: #0f172a;")
            self.check_ready()
            
    def check_ready(self):
        if self.input_file and self.output_dir:
            self.convert_btn.setEnabled(True)

    def log_message(self, msg):
        self.log_box.append(msg)

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def conversion_done(self, success, msg):
        self.convert_btn.setEnabled(True)
        self.in_btn.setEnabled(True)
        self.out_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Conversion Complete", "The model was successfully converted to diffusers format.")
        else:
            QMessageBox.critical(self, "Conversion Failed", f"An error occurred:\n{msg}")

    def start_conversion(self):
        if not self.input_file or not self.output_dir:
            return
            
        self.convert_btn.setEnabled(False)
        self.in_btn.setEnabled(False)
        self.out_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_box.clear()
        
        # Make a specific subdirectory for the model if not already specified
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        final_output_dir = os.path.join(self.output_dir, base_name)
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
            
        self.log_message(f"Destination folder initialized: {final_output_dir}")
        self.log_message("-" * 50)
        
        self.thread = ConverterThread(
            self.input_file, 
            final_output_dir, 
            self.half_check.isChecked(), 
            self.safetensors_check.isChecked()
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.log.connect(self.log_message)
        self.thread.finished.connect(self.conversion_done)
        self.thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SDXLConverterGUI()
    ex.show()
    sys.exit(app.exec())
