import tkinter as tk
import pandas as pd
from tkinter import ttk, scrolledtext
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import sys
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class ConversationHistory:
    def __init__(self):
        self.statements = []
        self.statuses = []
        self.confidences = []
        self.MAX_STATEMENTS = 5
        
    def add_entry(self, statement, status, confidence):
        self.statements.append(statement)
        self.statuses.append(status)
        self.confidences.append(confidence)
        
    def is_complete(self):
        return len(self.statements) >= self.MAX_STATEMENTS
        
    def get_final_diagnosis(self):
        if not self.is_complete():
            return None, None
          
        non_normal_statuses = [s for s in self.statuses if s != "Normal"]
        if not non_normal_statuses:
            primary_status = "Normal"  # If all were Normal
        else:
          status_counts = Counter(non_normal_statuses)
          primary_status = status_counts.most_common(1)[0][0]
        
        avg_confidence = sum(self.confidences) / len(self.confidences)
        return primary_status, avg_confidence
      
class LoadingWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Beacon - Your Mental Health Guide")
        
        width = 400
        height = 150
        self.root.geometry(f'{width}x{height}')
        
        # Center window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        
        # Loading message
        label = ttk.Label(self.root, text="Initializing Beacon...\nPlease wait", font=("Arial", 12))
        label.pack(expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate', length=44)
        self.progress.pack(fill='x', padx=20, pady=10)
        self.progress.start()

class MentalHealthApp:
    def __init__(self, root):
        self.root = root
        loading = LoadingWindow()
        loading.root.update()
        
        # Create main window but don't show it yet
        self.root.title("Beacon - Your Mental Health Guide")
        width = 800
        height = 600
        
        self.root.geometry(f'{width}x{height}')
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
      
        # Load models and prepare data
        self.load_models()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.predictor_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.predictor_tab, text='Predictor')
        self.notebook.add(self.analytics_tab, text='Analytics')
        
        self.conversation = ConversationHistory()

        # Create GUI elements
        self.create_predictor_widgets()
        self.create_analytics_widgets()
        
        # Show main window and destroy loading window
        loading.root.destroy()
        self.root.deiconify()  # Show main window

    def load_models(self):
        try:
            model_path = self.get_resource_path('model.pkl')
            vectorizer_path = self.get_resource_path('vectorizer.pkl')
            label_encoder_path = self.get_resource_path('label_encoder.pkl')
            data_path = self.get_resource_path('data.csv')
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(label_encoder_path, 'rb') as f:
                self.le = pickle.load(f)
                
            # Get predictions for visualization
            df = pd.read_csv(data_path)
            X = self.vectorizer.transform(df['statement'].fillna(''))
            y_true = self.le.transform(df['status'])
            y_pred = self.model.predict(X)
            y_pred = np.round(y_pred).astype(int)
            
            self.confusion_mat = confusion_matrix(y_true, y_pred)
            self.class_report = classification_report(y_true, y_pred, labels=np.unique(y_true), target_names=self.le.classes_,output_dict=True)
            self.class_distribution = pd.Series(df['status']).value_counts()
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Looking in: {self.get_resource_path('')}")
            sys.exit(1)
            
    def get_resource_path(self, relative_path):
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
      
    def create_predictor_widgets(self):
        # Chat container
        chat_container = ttk.Frame(self.predictor_tab)
        chat_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Chat history
        history_frame = ttk.LabelFrame(chat_container, text="Chat History")
        history_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.chat_history = scrolledtext.ScrolledText(
            history_frame, 
            wrap=tk.WORD,
            height=15,
            font=("Arial", 11),
            background="white"  # Set white background

        )
        self.chat_history.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_history.config(state='disabled')
        
         ##Configure message bubble styles with simulated rounded corners
        self.chat_history.tag_configure(
            "user", 
            background="#007AFF",  # iMessage blue
            foreground="white",
            lmargin1=200,  # Right align user messages
            lmargin2=200,
            rmargin=20,
            spacing1=10,  # Space above bubble
            spacing3=10,  # Space below bubble
            justify="right"
        )

        self.chat_history.tag_configure(
            "bot",
            background="#E9E9EB",  # iMessage gray
            foreground="black",
            lmargin1=20,   # Left align bot messages
            lmargin2=20,
            rmargin=200,
            spacing1=10,
            spacing3=10,
            justify="left"
        )

        self.chat_history.tag_configure(
            "diagnosis",
            background="#34C759",  # iOS green
            foreground="white",
            lmargin1=20,
            lmargin2=20,
            rmargin=200,
            spacing1=10,
            spacing3=10,
            font=("Arial", 11, "bold"),
            justify="left"
        )
        
        # Input area
        input_frame = ttk.Frame(chat_container)
        input_frame.pack(fill="x", pady=5)
        
        self.text_input = ttk.Entry(input_frame, font=("Arial", 11))
        self.text_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.text_input.bind("<Return>", lambda e: self.process_message())
        
        restart_btn = ttk.Button(
            input_frame, 
            text="Restart", 
            command=self.restart_chat
        )
        restart_btn.pack(side="right", padx=5)
        send_btn = ttk.Button(
            input_frame, 
            text="Send", 
            command=self.process_message
        )
        send_btn.pack(side="right")
        
        self.start_chat()
        

    def create_analytics_widgets(self):
        # Create notebook for analytics tabs
        analytics_notebook = ttk.Notebook(self.analytics_tab)
        analytics_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Confusion Matrix Tab
        confusion_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(confusion_tab, text='Confusion Matrix')
        
        fig1 = Figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        fig1.subplots_adjust(left=0.3, bottom=0.4)  # Increase left margin

        # Add labels to confusion matrix
        sns.heatmap(
            self.confusion_mat, 
            annot=True, 
            ax=ax1, 
            fmt='d',
            xticklabels=self.le.classes_,
            yticklabels=self.le.classes_
        )        
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        canvas1 = FigureCanvasTkAgg(fig1, confusion_tab)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Classification Report Tab
        metrics_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(metrics_tab, text='Classification Metrics')
        
        report_df = pd.DataFrame(self.class_report)
        if 'support' in report_df.index:
            report_df = report_df.drop('support', axis=0)
        
        fig2 = Figure(figsize=(8, 6))
        fig2.subplots_adjust(left=0.25)  # Increase left margin
        ax2 = fig2.add_subplot(111)
        sns.heatmap(report_df.T, annot=True, ax=ax2, fmt='.2f')
        ax2.set_title('Classification Metrics')
        canvas2 = FigureCanvasTkAgg(fig2, metrics_tab)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Distribution Tab
        dist_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(dist_tab, text='Class Distribution')
        
        fig3 = Figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111)
        ax3.pie(self.class_distribution.values, labels=self.class_distribution.index, autopct='%1.1f%%')
        ax3.set_title('Class Distribution')
        canvas3 = FigureCanvasTkAgg(fig3, dist_tab)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def process_message(self):
        message = self.text_input.get().strip()
        if not message:
            return
            
        self.text_input.delete(0, tk.END)
        self.add_message(f"You: {message}", "user")
        
        # Process with model
        X = self.vectorizer.transform([message])
        prediction = self.model.predict_proba(X)[0]
        status = self.le.inverse_transform([prediction.argmax()])[0]
        confidence = prediction.max()
        
        self.conversation.add_entry(message, status, confidence)
        
        if status != 'Normal':
          response = "It sounds like you may be feeling " + status + "."
          self.add_message(
              f"Bot: {response} (confidence: {confidence:.2f})",
              "bot"
          )
        
        if self.conversation.is_complete():
            self.show_final_diagnosis()
        else:
            next_question = self.get_follow_up_question()
            self.add_message(f"Bot: {next_question}", "bot")

    def add_message(self, message, tag):
        self.chat_history.config(state='normal')
        
        # Insert message
        start_index = self.chat_history.index("end-1c")
        self.chat_history.insert(tk.END, message)
        
        # Add extra newline after message
        self.chat_history.insert(tk.END, "\n\n")
        
        # Apply tag to each line of the message
        lines = message.split('\n')
        current_index = start_index
        
        for line in lines:
            if line.strip():  # Skip empty lines
                line_start = current_index
                line_end = self.chat_history.index(f"{line_start} lineend")
                self.chat_history.tag_add(tag, line_start, line_end)
            current_index = self.chat_history.index(f"{current_index} lineend+1c")
        
        self.chat_history.see(tk.END)
        self.chat_history.config(state='disabled')

    def get_follow_up_question(self):
      general_questions = [
            "Can you tell me more about how you're feeling?",
            "Could you describe what's been going through your mind lately?",
            "How have these feelings been affecting your daily life?",
            "Can you tell me more about how you're feeling?",
            "What do you think triggered these emotions?"
        ]
      return general_questions[len(self.conversation.statements) - 1]

    def start_chat(self):
        welcome = "Hi! I'm Beacon. I'm here to help you with your mental health. How are you feeling today?"
        self.add_message(f"Bot: {welcome}", "bot")
    
    def restart_chat(self):
      # Clear conversation history object
      self.conversation = ConversationHistory()
      
      # Clear chat display
      self.chat_history.config(state='normal')
      self.chat_history.delete('1.0', tk.END)
      self.chat_history.config(state='disabled')
      
      # Clear input field
      self.text_input.delete(0, tk.END)
      
      # Start new chat
      self.start_chat()
        
    def show_final_diagnosis(self):
        status, confidence = self.conversation.get_final_diagnosis()
        if status == "Normal":
            summary = f"You seem to be feeling okay. If you have any concerns, please consult a professional. Overall Confidence: {confidence:.2f}"
        else:
            summary = f"Possible Diagnosis: {status}, Overall Confidence: {confidence:.2f}"
        self.add_message(summary, "diagnosis")
        self.add_message(self.get_resources_message(), "bot")
        self.add_message("Would you like to restart the conversation? Use the restart button to do so.", "bot")

    def get_resources_message(self):
        message = "Here are some resources that may help you:\n"
        general_resources = [
          "Anxiety and Depression Association of America: https://adaa.org/",
          "National Alliance on Mental Illness: https://www.nami.org/",
          "Crisis Text Line: https://www.crisistextline.org/",
          "Mental Health America: https://www.mhanational.org/",
        ]
        therapy_resources = [
          "BetterHelp: https://www.betterhelp.com/",
          "Talkspace: https://www.talkspace.com/",
          "Psychology Today: https://www.psychologytoday.com/us"
        ]
        self_help_resources = [
          "Headspace: https://www.headspace.com/",
          "Calm: https://www.calm.com/",
          "Calm Clinic: https://www.calmclinic.com/"
        ]
        crisis_resources = [
          "Crisis Text Line: Text \"HOME\" to 741741 or go to https://www.crisistextline.org/",
          "Suicide Prevention Lifeline: 1-800-273-8255",
        ]
        message += "General Resources:\n" + "\n".join(general_resources) + "\n\n"
        message += "Therapy Resources:\n" + "\n".join(therapy_resources) + "\n\n"
        message += "Self-Help Resources:\n" + "\n".join(self_help_resources) + "\n\n"
        message += "Crisis Resources:\n" + "\n".join(crisis_resources)
        return message
        

def main():
    root = tk.Tk()
    root.withdraw() 
    app = MentalHealthApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()