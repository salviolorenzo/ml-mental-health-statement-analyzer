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

class MentalHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental Health Statement Analyzer")
        self.root.geometry("800x600")
      
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
    
    def load_models(self):
        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                self.le = pickle.load(f)
            with open('status_labels.pkl', 'rb') as f:
                self.status_labels = pickle.load(f)
            
            # Get predictions for visualization
            df = pd.read_csv('data.csv')
            X = self.vectorizer.transform(df['statement'].fillna(''))
            y_true = self.le.transform(df['status'])
            y_pred = self.model.predict(X)
            y_pred = np.round(y_pred).astype(int)
            
            self.confusion_mat = confusion_matrix(y_true, y_pred)
            self.class_report = classification_report(y_true, y_pred, labels=np.unique(y_true), target_names=self.le.classes_,output_dict=True)
            self.class_distribution = pd.Series(df['status']).value_counts()
            
        except FileNotFoundError:
            print("Model files not found.")
            self.root.destroy()
    
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
            font=("Arial", 11)
        )
        self.chat_history.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_history.config(state='disabled')
        
        # Configure chat styles
        self.chat_history.tag_configure(
            "user", 
            background="#e3f2fd",
            foreground="black",
            lmargin1=20,
            rmargin=20
        )
        self.chat_history.tag_configure(
            "bot",
            background="#f5f5f5",
            foreground="black",
            lmargin1=20,
            rmargin=20
        )
        self.chat_history.tag_configure(
            "diagnosis",
            background="#e8f5e9",
            foreground="black",
            font=("Arial", 11, "bold"),
            lmargin1=20,
            rmargin=20
        )
        
        # Input area
        input_frame = ttk.Frame(chat_container)
        input_frame.pack(fill="x", pady=5)
        
        self.text_input = ttk.Entry(input_frame, font=("Arial", 11))
        self.text_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.text_input.bind("<Return>", lambda e: self.process_message())
        
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
        self.chat_history.insert(tk.END, message + "\n\n")
        last_line_start = self.chat_history.index("end-3c linestart")
        last_line_end = self.chat_history.index("end-1c")
        self.chat_history.tag_add(tag, last_line_start, last_line_end)
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
        welcome = "Hi! I'm here to understand how you're feeling. Please share your thoughts with me."
        self.add_message(f"Bot: {welcome}", "bot")
        
    def show_final_diagnosis(self):
        status, confidence = self.conversation.get_final_diagnosis()
        if status == "Normal":
            summary = f"You seem to be feeling okay. If you have any concerns, please consult a professional. Overall Confidence: {confidence:.2f}"
        else:
          summary = f"Possible Diagnosis: {status}, Overall Confidence: {confidence:.2f}"
        self.add_message(summary, "diagnosis")

def main():
    root = tk.Tk()
    app = MentalHealthApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()