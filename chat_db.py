from peewee import *                                                    # Import the required modules
from datetime import datetime                                           # Import the required modules


db = SqliteDatabase('chat_history.db')                                  # Create a database object

class BaseModel(Model):                                                 # Base model for the database
    class Meta:                                                         # Meta class for the base model          
        database = db                                                   # Set the database for the model

class Conversation(BaseModel):
    id = AutoField()                                                    # Unique ID for the conversation
    conversation_id = CharField(unique=True)                            # The ID of the conversation
    start_time = DateTimeField(default=datetime.now)                    # The time the conversation was started
    last_update = DateTimeField(default=datetime.now)                   # The time the conversation was last updated

class Message(BaseModel):                                               # A message sent in a conversation
    id = AutoField()                                                    # Unique ID for the message
    conversation = ForeignKeyField(Conversation, backref='messages')    # The conversation this message belongs to
    content = TextField()                                               # The content of the message
    is_user = BooleanField()                                            # True if the message is from the user, False if from the bot
    timestamp = DateTimeField(default=datetime.now)                     # The time the message was sent

def initialize_db():                                                    # Create the tables in the database
    db.connect()                                                        # Connect to the database
    db.create_tables([Conversation, Message])                           # Create the tables
    db.close()                                                          # Close the connection                                                
