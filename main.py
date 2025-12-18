from routing_engine import RoutingEngine

def main():
    engine = RoutingEngine()

    while True:
        query = input("\nEnter a query (or 'exit'): ")
        if query.lower() == "exit":
            break

        response = engine.run(query)
        print("\n--- RESPONSE ---")
        print(response.content)

if __name__ == "__main__":
    main()
