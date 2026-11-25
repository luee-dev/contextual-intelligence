from sqlalchemy import text

from contextual_intelligence.database.connection import engine


def main():
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database(), current_user")
            )

            database, user = result.fetchone()

            print("Database connection successful!")
            print(f"Database: {database}")
            print(f"User: {user}")

    except Exception as error:
        print(f"Database connection failed: {error}")
        raise


if __name__ == "__main__":
    main()