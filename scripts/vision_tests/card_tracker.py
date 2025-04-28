# Use this script to track the cards detected in a game of blackjack.
# This script is designed to work with YOLOv11 and OpenCV for real-time card detection.

class CardTracker:
    SUITS = {"Spades": "S", "Hearts": "H", "Diamonds": "D", "Clubs": "C"}
    FACE_CARDS = {"Ace": "A", "Jack": "J", "Queen": "Q", "King": "K"}

    def __init__(self, all_cards, min_frames=5):
        """
        Initializes the CardTracker class.

        :param all_cards: List of all possible card names in the deck.
        :param min_frames: Number of frames a card must be seen before printing.
        """
        self.full_deck = set(self.convert_card_names(all_cards))  # Convert to YOLO-compatible format
        self.detected_cards = set()  # Cards that have been seen
        self.seen_cards = {}  # Dictionary to track detection count
        self.min_frames = min_frames

    def convert_card_names(self, card_list):
        """
        Converts full card names (e.g., '3 of Diamonds') to YOLO-compatible shorthand ('3D').
        Converts face cards and aces correctly (e.g., 'King of Hearts' → 'KH').
        """
        converted_cards = set()
        for name in card_list:
            parts = name.split()
            rank = parts[0]
            suit = self.SUITS[parts[-1]]  # Convert suit names

            if rank in self.FACE_CARDS:
                shorthand = f"{self.FACE_CARDS[rank]}{suit}"  # Convert face cards and aces
            else:
                shorthand = f"{rank}{suit}"  # Convert numbered cards normally

            converted_cards.add(shorthand)
        return converted_cards

    def update(self, detected_cards):
        """
        Updates the tracker with detected cards and determines when to print.

        :param detected_cards: List of detected card names in the current frame.
        """
        detected_cards = {card.upper() for card in detected_cards}  # Ensure uniform naming
        current_seen = set()

        for card in detected_cards:
            if card in self.seen_cards:
                self.seen_cards[card] += 1
            else:
                self.seen_cards[card] = 1

            # Print the card and mark it as detected only if it has been seen in 5 different frames
            if self.seen_cards[card] >= self.min_frames:
                if card not in self.detected_cards:
                    print(f"Detected Card: {card}")
                    # self.detected_cards.add(card)  # Ensure the card is marked as detected

            current_seen.add(card)

        # Ensure unseen cards don’t reset but don’t increase count either
        for card in list(self.seen_cards.keys()):
            if card not in current_seen:
                self.seen_cards[card] = max(0, self.seen_cards[card] - 1)  # Reduce count but do not erase

        # Debugging print to verify which cards have been fully detected
        # print(f"DEBUG - Fully Detected Cards So Far: {sorted(self.detected_cards)}")

    def get_remaining_cards(self):
        """
        Returns a list of cards that have not been detected yet.
        """
        return list(self.full_deck - self.detected_cards)

    def reset(self):
        """
        Resets the detected cards and seen tracking, allowing a fresh start.
        """
        self.detected_cards.clear()
        self.seen_cards.clear()
        print("Deck reset! Dictionary of seen cards has been cleared.")

