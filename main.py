from fastapi import FastAPI

app=FastAPI(name="langgraph-ai-agent")

@app.get("/health")
async def health_check():
    return {"Status":"ok"}

def main():
    print("Hello from Chatterbot")
    
if __name__ == "__main__":
    main()

