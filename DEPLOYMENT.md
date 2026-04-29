# Execution Guide: Web-Tool

This guide details the necessary steps to set up both the backend and frontend of the application.

> [!NOTE]
> Do not forget to install the dependencies listed in **requirements.txt** using pip.

## 1. Backend (FastAPI)

The backend manages business logic and the database through an API built with FastAPI.

### Prerequisites
* Have the Python **virtual environment** for the project active.

### Deployment Steps
1.  **Navigate to the backend folder:**
    ```bash
    cd web-tool/backend
    ```
2.  **Install project dependencies:**
    Ensure your virtual environment is active and install the required libraries specified in the root file:
    ```bash
    pip install -r ../../requirements.txt
    ```
3.  **Launch the API server:**
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    ```

* **API available at:** [http://localhost:8000](http://localhost:8000)

---

## 2. Frontend (Next.js)

The frontend is the user interface built with the Next.js framework.

### Prerequisites
* **Node.js** installed on your system.

### Deployment Steps
1.  **Navigate to the frontend folder:**
    ```bash
    cd web-tool/frontend
    ```
2.  **Install Node dependencies:**
    ```bash
    npm install
    ```
3.  **Launch the frontend in development mode:**
    ```bash
    npm run dev
    ```

* **Frontend available at:** [http://localhost:3000](http://localhost:3000)

---

## 💡 Important Notes

> [!IMPORTANT]
> **API Configuration:** The frontend is configured to consume the API at `http://localhost:8000/api`. Therefore, it is essential that the backend is running on port **8000**.

> [!TIP]
> **Simultaneous Execution:** To work with both parts at once, you should open **two terminals** or console tabs: one dedicated to the backend and another for the frontend.