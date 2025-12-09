import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from datetime import datetime
import json

class GestureRecognizer:
    """Advanced gesture recognition for medical interface."""
    
    def __init__(self):
        """Initialize gesture recognizer."""
        self.gesture_history = deque(maxlen=5)
        self.current_gesture = "NONE"
        self.gesture_confidence = 0.0
        self.last_gesture_time = time.time()
        self.gesture_hold_duration = 0.0
    
    def count_extended_fingers(self, hand_landmarks):
        """Count number of extended fingers."""
        fingers = []
        
        # Thumb (check horizontal distance)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        if abs(thumb_tip.x - thumb_ip.x) > 0.04:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (check if tip is above pip joint)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y - 0.02:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers), fingers
    
    def detect_pinch(self, hand_landmarks):
        """Detect pinch gesture (thumb + index close)."""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        return distance < 0.05
    
    def detect_thumbs_up(self, hand_landmarks):
        """Detect thumbs up gesture."""
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        
        # Thumb pointing up
        thumb_up = thumb_tip.y < thumb_mcp.y - 0.1
        
        # Other fingers closed
        finger_count, _ = self.count_extended_fingers(hand_landmarks)
        
        return thumb_up and finger_count <= 2
    
    def detect_thumbs_down(self, hand_landmarks):
        """Detect thumbs down gesture."""
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Thumb pointing down
        thumb_down = thumb_tip.y > thumb_mcp.y + 0.1
        
        # Other fingers closed
        finger_count, _ = self.count_extended_fingers(hand_landmarks)
        
        return thumb_down and finger_count <= 2
    
    def detect_peace_sign(self, hand_landmarks):
        """Detect peace sign (index + middle extended)."""
        finger_count, fingers = self.count_extended_fingers(hand_landmarks)
        
        # Index and middle extended, others closed
        return finger_count == 2 and fingers[1] == 1 and fingers[2] == 1
    
    def detect_pointing(self, hand_landmarks):
        """Detect pointing gesture (only index extended)."""
        finger_count, fingers = self.count_extended_fingers(hand_landmarks)
        
        return finger_count == 1 and fingers[1] == 1
    
    def detect_open_palm(self, hand_landmarks):
        """Detect open palm (all fingers extended)."""
        finger_count, _ = self.count_extended_fingers(hand_landmarks)
        return finger_count >= 4
    
    def detect_fist(self, hand_landmarks):
        """Detect closed fist."""
        finger_count, _ = self.count_extended_fingers(hand_landmarks)
        return finger_count == 0
    
    def get_hand_position(self, hand_landmarks):
        """Get normalized hand position."""
        wrist = hand_landmarks.landmark[0]
        return wrist.x, wrist.y
    
    def recognize(self, hand_landmarks):
        """
        Recognize gesture from hand landmarks.
        
        Returns:
            tuple: (gesture_name, confidence, metadata)
        """
        current_time = time.time()
        
        # Detect all gestures
        is_pinch = self.detect_pinch(hand_landmarks)
        is_peace = self.detect_peace_sign(hand_landmarks)
        is_pointing = self.detect_pointing(hand_landmarks)
        is_thumbs_up = self.detect_thumbs_up(hand_landmarks)
        is_thumbs_down = self.detect_thumbs_down(hand_landmarks)
        is_open = self.detect_open_palm(hand_landmarks)
        is_fist = self.detect_fist(hand_landmarks)
        finger_count, _ = self.count_extended_fingers(hand_landmarks)
        
        # Priority-based gesture recognition
        gesture = "NONE"
        confidence = 0.0
        metadata = {}
        
        if is_pinch:
            gesture = "PINCH"
            confidence = 0.95
        elif is_peace:
            gesture = "PEACE"
            confidence = 0.9
        elif is_thumbs_up:
            gesture = "THUMBS_UP"
            confidence = 0.9
        elif is_thumbs_down:
            gesture = "THUMBS_DOWN"
            confidence = 0.9
        elif is_pointing:
            gesture = "POINTING"
            confidence = 0.85
            x, y = self.get_hand_position(hand_landmarks)
            metadata['position'] = (x, y)
        elif is_open:
            gesture = "OPEN_PALM"
            confidence = 0.8
        elif is_fist:
            gesture = "FIST"
            confidence = 0.8
        elif 1 <= finger_count <= 5:
            gesture = f"FINGERS_{finger_count}"
            confidence = 0.75
            metadata['count'] = finger_count
        
        # Add to history for smoothing
        self.gesture_history.append((gesture, confidence))
        
        # Get most common recent gesture
        if len(self.gesture_history) >= 3:
            recent_gestures = [g[0] for g in self.gesture_history]
            most_common = max(set(recent_gestures), key=recent_gestures.count)
            
            # Calculate hold duration
            if most_common == self.current_gesture:
                self.gesture_hold_duration = current_time - self.last_gesture_time
            else:
                self.current_gesture = most_common
                self.last_gesture_time = current_time
                self.gesture_hold_duration = 0.0
            
            gesture = most_common
        
        self.gesture_confidence = confidence
        metadata['hold_duration'] = self.gesture_hold_duration
        
        return gesture, confidence, metadata


class Notification:
    """Visual notification for gestures."""
    
    def __init__(self, message, notification_type="info", duration=3.0):
        """Initialize notification."""
        self.message = message
        self.type = notification_type  # info, success, warning, error, critical
        self.duration = duration
        self.start_time = time.time()
        self.alpha = 1.0
    
    def is_active(self):
        """Check if notification is still active."""
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            return False
        
        # Fade out in last 0.5 seconds
        if elapsed > self.duration - 0.5:
            self.alpha = (self.duration - elapsed) / 0.5
        
        return True
    
    def draw(self, frame, y_offset=0):
        """Draw notification on frame."""
        h, w = frame.shape[:2]
        
        # Color based on type
        colors = {
            'info': (255, 200, 100),
            'success': (100, 255, 100),
            'warning': (100, 200, 255),
            'error': (100, 100, 255),
            'critical': (0, 0, 255)
        }
        color = colors.get(self.type, (255, 255, 255))
        
        # Calculate size
        padding = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(self.message, font, font_scale, thickness)[0]
        box_width = text_size[0] + padding * 2
        box_height = text_size[1] + padding * 2
        
        # Position (top center)
        x = (w - box_width) // 2
        y = 80 + y_offset
        
        # Draw with alpha
        overlay = frame.copy()
        
        # Background
        bg_color = (40, 40, 40)
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), bg_color, -1)
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), color, 3)
        
        # Text
        text_x = x + padding
        text_y = y + padding + text_size[1]
        cv2.putText(overlay, self.message, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Blend with alpha
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)


class NotificationManager:
    """Manages multiple notifications."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.notifications = []
        self.max_visible = 1  # Only show 1 notification at a time
    
    def add(self, message, notification_type="info", duration=3.0):
        """Add a notification (replaces old ones)."""
        # Clear existing notifications when adding new one
        self.notifications = []
        self.notifications.append(Notification(message, notification_type, duration))
    
    def update_and_draw(self, frame):
        """Update and draw all active notifications."""
        # Remove inactive notifications
        self.notifications = [n for n in self.notifications if n.is_active()]
        
        # Only show the most recent notification
        if self.notifications:
            self.notifications[-1].draw(frame, y_offset=0)


class PatientActionHandler:
    """Handles patient-specific actions."""
    
    def __init__(self):
        """Initialize patient action handler."""
        self.last_action_time = {}
        self.action_cooldown = 5.0  # Increased to 5 seconds to prevent spam
        self.action_log = []
        self.pain_level = 0
        self.active_actions = set()  # Track currently active actions
    
    def can_perform_action(self, action_name):
        """Check if action can be performed (cooldown)."""
        current_time = time.time()
        last_time = self.last_action_time.get(action_name, 0)
        
        # Check if still in cooldown
        if current_time - last_time < self.action_cooldown:
            return False
        
        # Check if action is already active
        if action_name in self.active_actions:
            return False
        
        return True
    
    def log_action(self, action_name, details=None):
        """Log an action with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'time': timestamp,
            'action': action_name,
            'details': details
        }
        self.action_log.append(log_entry)
        self.last_action_time[action_name] = time.time()
        self.active_actions.add(action_name)
        
        # Keep only last 20 actions
        if len(self.action_log) > 20:
            self.action_log.pop(0)
    
    def clear_active_action(self, action_name):
        """Clear an active action after cooldown."""
        if action_name in self.active_actions:
            self.active_actions.discard(action_name)
    
    def handle_gesture(self, gesture, metadata, notification_manager):
        """Handle patient gesture and trigger appropriate action."""
        action_taken = None
        
        # Require gesture to be held for at least 1 second
        hold_duration = metadata.get('hold_duration', 0)
        
        if gesture == "PEACE" and hold_duration > 1.0:
            if self.can_perform_action("call_nurse"):
                self.log_action("Call Nurse")
                notification_manager.add("🔔 Nurse Called - Help is on the way!", "success", 4.0)
                action_taken = "NURSE_CALLED"
        
        elif gesture.startswith("FINGERS_") and hold_duration > 1.0:
            finger_count = metadata.get('count', 0)
            if 1 <= finger_count <= 5:
                action_key = f"pain_level_{finger_count}"
                if self.can_perform_action(action_key):
                    self.pain_level = finger_count
                    self.log_action("Pain Level Report", f"Level {finger_count}/5")
                    
                    pain_messages = {
                        1: "😊 Pain Level 1 - Minimal discomfort",
                        2: "🙂 Pain Level 2 - Mild pain",
                        3: "😐 Pain Level 3 - Moderate pain",
                        4: "😣 Pain Level 4 - Severe pain",
                        5: "😱 Pain Level 5 - Extreme pain - Notifying staff!"
                    }
                    
                    msg_type = "critical" if finger_count >= 4 else "warning"
                    notification_manager.add(pain_messages[finger_count], msg_type, 5.0)
                    action_taken = f"PAIN_LEVEL_{finger_count}"
        
        elif gesture == "PINCH" and hold_duration > 1.0:
            if self.can_perform_action("request_water"):
                self.log_action("Request Water")
                notification_manager.add("💧 Water Requested - Coming soon!", "info", 3.0)
                action_taken = "WATER_REQUESTED"
        
        elif gesture == "THUMBS_UP" and hold_duration > 1.0:
            if self.can_perform_action("feeling_good"):
                self.log_action("Feeling Good")
                notification_manager.add("👍 Great! Glad you're doing well!", "success", 3.0)
                action_taken = "POSITIVE_FEEDBACK"
        
        elif gesture == "THUMBS_DOWN" and hold_duration > 1.0:
            if self.can_perform_action("need_help"):
                self.log_action("Need Assistance")
                notification_manager.add("⚠️ Assistance needed - Staff notified", "warning", 4.0)
                action_taken = "ASSISTANCE_NEEDED"
        
        elif gesture == "OPEN_PALM" and hold_duration > 2.5:
            if self.can_perform_action("emergency"):
                self.log_action("EMERGENCY", "Both hands raised")
                notification_manager.add("🚨 EMERGENCY ALERT - Medical team responding!", "critical", 5.0)
                action_taken = "EMERGENCY"
        
        return action_taken


class DoctorActionHandler:
    """Handles doctor-specific actions."""
    
    def __init__(self):
        """Initialize doctor action handler."""
        self.last_action_time = {}
        self.action_cooldown = 2.0  # Increased cooldown
        self.current_patient_record = 1
        self.zoom_level = 1.0
        self.marked_points = []
        self.active_actions = set()
    
    def can_perform_action(self, action_name):
        """Check if action can be performed."""
        current_time = time.time()
        last_time = self.last_action_time.get(action_name, 0)
        
        if current_time - last_time < self.action_cooldown:
            return False
        
        if action_name in self.active_actions:
            return False
        
        return True
    
    def handle_gesture(self, gesture, metadata, notification_manager):
        """Handle doctor gesture and trigger appropriate action."""
        action_taken = None
        hold_duration = metadata.get('hold_duration', 0)
        
        if gesture == "POINTING" and hold_duration > 1.0:
            if self.can_perform_action("mark_area"):
                pos = metadata.get('position', (0, 0))
                self.marked_points.append(pos)
                notification_manager.add(f"📍 Area Marked", "info", 2.0)
                action_taken = "AREA_MARKED"
                self.active_actions.add("mark_area")
                self.last_action_time["mark_area"] = time.time()
        
        elif gesture == "PINCH" and hold_duration > 1.5:
            if self.can_perform_action("zoom"):
                self.zoom_level = min(4.0, self.zoom_level + 0.5)  # Max 4x zoom
                notification_manager.add(f"🔍 Zoom: {self.zoom_level:.1f}x - Hold to zoom more", "info", 2.0)
                action_taken = "ZOOM_IN"
                self.active_actions.add("zoom")
                self.last_action_time["zoom"] = time.time()
        
        elif gesture == "OPEN_PALM" and hold_duration > 1.0:
            if self.can_perform_action("zoom_out"):
                self.zoom_level = max(1.0, self.zoom_level - 0.5)
                if self.zoom_level == 1.0:
                    notification_manager.add(f"🔍 Zoom Reset - Normal View", "info", 2.0)
                else:
                    notification_manager.add(f"🔍 Zoom: {self.zoom_level:.1f}x", "info", 2.0)
                action_taken = "ZOOM_OUT"
                self.active_actions.add("zoom_out")
                self.last_action_time["zoom_out"] = time.time()
        
        elif gesture == "THUMBS_UP" and hold_duration > 1.0:
            if self.can_perform_action("approve"):
                notification_manager.add("✅ Approved", "success", 3.0)
                action_taken = "APPROVED"
                self.active_actions.add("approve")
                self.last_action_time["approve"] = time.time()
        
        elif gesture == "THUMBS_DOWN" and hold_duration > 1.0:
            if self.can_perform_action("reject"):
                notification_manager.add("❌ Rejected", "error", 3.0)
                action_taken = "REJECTED"
                self.active_actions.add("reject")
                self.last_action_time["reject"] = time.time()
        
        elif gesture == "PEACE" and hold_duration > 1.0:
            if self.can_perform_action("next_patient"):
                self.current_patient_record += 1
                notification_manager.add(f"📄 Patient Record #{self.current_patient_record}", "info", 3.0)
                action_taken = "NEXT_RECORD"
                self.active_actions.add("next_patient")
                self.last_action_time["next_patient"] = time.time()
        
        elif gesture == "FIST" and hold_duration > 1.0:
            if self.can_perform_action("clear_marks"):
                self.marked_points = []
                notification_manager.add("🗑️ Marks Cleared", "info", 2.0)
                action_taken = "CLEARED"
                self.active_actions.add("clear_marks")
                self.last_action_time["clear_marks"] = time.time()
        
        return action_taken




class HandTracker:
    """MediaPipe hand tracking."""
    
    def __init__(self):
        """Initialize MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def process_frame(self, frame):
        """Process frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)
    
    def draw_landmarks(self, frame, hand_landmarks, color=(0, 255, 255)):
        """Draw hand landmarks."""
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
    
    def close(self):
        """Release resources."""
        self.hands.close()


class GestureMedApp:
    """GestureMed Hospital Interface System."""
    
    def __init__(self):
        """Initialize application."""
        # Components
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.notification_manager = NotificationManager()
        self.patient_handler = PatientActionHandler()
        self.doctor_handler = DoctorActionHandler()
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # State
        self.mode = "PATIENT"  # PATIENT or DOCTOR
        self.running = True
        self.show_help = True
        self.show_action_log = True
        
        # FPS
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Welcome message
        self.notification_manager.add("Welcome to GestureMed System", "success", 5.0)
    
    def switch_mode(self):
        """Switch between Patient and Doctor modes."""
        self.mode = "DOCTOR" if self.mode == "PATIENT" else "PATIENT"
        mode_name = "Doctor Mode" if self.mode == "DOCTOR" else "Patient Mode"
        self.notification_manager.add(f"Switched to {mode_name}", "info", 3.0)
    
    def draw_ui(self, frame):
        """Draw main UI elements."""
        h, w = frame.shape[:2]
        
        # Mode indicator (top left)
        mode_color = (100, 200, 255) if self.mode == "PATIENT" else (255, 150, 100)
        mode_icon = "🤒" if self.mode == "PATIENT" else "👨‍⚕️"
        mode_text = f"{mode_icon} {self.mode} MODE"
        
        cv2.rectangle(frame, (10, 10), (300, 70), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (300, 70), mode_color, 3)
        cv2.putText(frame, mode_text, (25, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2, cv2.LINE_AA)
        
        # Current gesture display (top right)
        gesture = self.gesture_recognizer.current_gesture
        confidence = self.gesture_recognizer.gesture_confidence
        
        if gesture != "NONE":
            gesture_display = gesture.replace("_", " ")
            gesture_text = f"✋ {gesture_display}"
            conf_text = f"{int(confidence * 100)}%"
            
            text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            box_width = max(text_size[0] + 40, 200)
            
            cv2.rectangle(frame, (w - box_width - 10, 10), (w - 10, 70), (20, 20, 20), -1)
            cv2.rectangle(frame, (w - box_width - 10, 10), (w - 10, 70), (100, 255, 100), 3)
            cv2.putText(frame, gesture_text, (w - box_width + 10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(frame, conf_text, (w - box_width + 10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Zoom indicator (if in doctor mode and zoomed)
        if self.mode == "DOCTOR" and self.doctor_handler.zoom_level > 1.0:
            zoom_text = f"🔍 {self.doctor_handler.zoom_level:.1f}x ZOOM"
            zoom_color = (100, 200, 255)
            
            cv2.rectangle(frame, (10, 80), (200, 130), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 80), (200, 130), zoom_color, 3)
            cv2.putText(frame, zoom_text, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, zoom_color, 2, cv2.LINE_AA)
        
        # Action log (bottom left)
        if self.show_action_log:
            log_entries = self.patient_handler.action_log[-5:] if self.mode == "PATIENT" else []
            
            if log_entries:
                log_y = h - 200
                cv2.rectangle(frame, (10, log_y), (350, h - 10), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, log_y), (350, h - 10), (200, 200, 200), 2)
                
                cv2.putText(frame, "RECENT ACTIONS:", (20, log_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                y = log_y + 50
                for entry in reversed(log_entries):
                    action_text = f"{entry['time']} - {entry['action']}"
                    cv2.putText(frame, action_text, (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                    y += 25
        
        # Help panel (bottom right)
        if self.show_help:
            self.draw_help_panel(frame)
        
        # FPS counter
        fps = int(np.mean(self.fps_history)) if len(self.fps_history) > 0 else 0
        cv2.putText(frame, f"FPS: {fps}", (w - 100, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
    
    def draw_help_panel(self, frame):
        """Draw help panel with gesture guide."""
        h, w = frame.shape[:2]
        
        if self.mode == "PATIENT":
            help_lines = [
                "PATIENT GESTURES:",
                "Peace (2) - Call Nurse",
                "1-5 Fingers - Pain Level",
                "Pinch - Request Water",
                "Thumbs Up - Feeling Good",
                "Thumbs Down - Need Help",
                "Open Palm (hold) - Emergency",
            ]
        else:
            help_lines = [
                "DOCTOR GESTURES:",
                "Point - Mark Area",
                "Pinch - Zoom In",
                "Open Palm - Zoom Out",
                "Thumbs Up - Approve",
                "Thumbs Down - Reject",
                "Peace - Next Record",
                "Fist - Clear Marks",
            ]
        
        panel_height = len(help_lines) * 25 + 40
        panel_width = 280
        
        y_start = h - panel_height - 10
        x_start = w - panel_width - 10
        
        # Background
        cv2.rectangle(frame, (x_start, y_start), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (x_start, y_start), (w - 10, h - 10), (150, 150, 150), 2)
        
        # Draw lines
        y = y_start + 30
        for i, line in enumerate(help_lines):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            font_scale = 0.5 if i == 0 else 0.4
            cv2.putText(frame, line, (x_start + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            y += 25
        
        # Controls at bottom
        cv2.putText(frame, "M-Mode H-Help Q-Quit", (x_start + 15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    
    def update(self):
        """Main update loop."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        # Process hand tracking
        results = self.hand_tracker.process_frame(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                color = (100, 200, 255) if self.mode == "PATIENT" else (255, 150, 100)
                self.hand_tracker.draw_landmarks(frame, hand_landmarks, color)
                
                # Recognize gesture
                gesture, confidence, metadata = self.gesture_recognizer.recognize(hand_landmarks)
                
                # Handle gesture based on mode
                if self.mode == "PATIENT":
                    self.patient_handler.handle_gesture(gesture, metadata, self.notification_manager)
                else:
                    self.doctor_handler.handle_gesture(gesture, metadata, self.notification_manager)
        
        # Draw UI
        self.draw_ui(frame)
        
        # Update and draw notifications
        self.notification_manager.update_and_draw(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if current_time != self.last_frame_time else 0
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        return frame
    
    def run(self):
        """Main loop."""
        print("=" * 80)
        print("🏥 GESTUREMED - CONTACTLESS HOSPITAL INTERFACE SYSTEM")
        print("=" * 80)
        print("\n✨ Revolutionary gesture-based control for healthcare environments")
        print("   Reduces infection spread by eliminating touchpoints")
        print("\n📋 PATIENT MODE Gestures:")
        print("   ✌️  Peace Sign      → Call Nurse")
        print("   ✋ Show 1-5 Fingers → Report Pain Level (1=mild, 5=severe)")
        print("   🤏 Pinch           → Request Water")
        print("   👍 Thumbs Up       → Feeling Good")
        print("   👎 Thumbs Down     → Need Assistance")
        print("   🙌 Open Palm (hold) → EMERGENCY ALERT")
        print("\n👨‍⚕️ DOCTOR MODE Gestures:")
        print("   👆 Point          → Mark Critical Area")
        print("   🤏 Pinch (hold)   → Zoom In on Image")
        print("   ✋ Open Palm      → Zoom Out")
        print("   👍 Thumbs Up      → Approve/Confirm")
        print("   👎 Thumbs Down    → Reject/Decline")
        print("   ✌️  Peace Sign     → Next Patient Record")
        print("   ✊ Fist           → Clear Markings")
        print("\n⌨️  CONTROLS:")
        print("   M - Switch Mode (Patient ↔ Doctor)")
        print("   H - Toggle Help Panel")
        print("   Q or ESC - Quit Application")
        print("\n💡 TIP: Hold gestures steady for 0.5s for best recognition")
        print("=" * 80)
        print("\nStarting GestureMed System...\n")
        
        while self.running:
            frame = self.update()
            
            if frame is not None:
                cv2.imshow('GestureMed - Contactless Hospital Interface', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
            elif key == ord('m'):
                self.switch_mode()
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('l'):
                self.show_action_log = not self.show_action_log
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 SESSION SUMMARY")
        print("=" * 80)
        print(f"Total Actions Logged: {len(self.patient_handler.action_log)}")
        if self.mode == "PATIENT" and self.patient_handler.action_log:
            print("\nRecent Patient Actions:")
            for entry in self.patient_handler.action_log[-5:]:
                details = f" - {entry['details']}" if entry.get('details') else ""
                print(f"  [{entry['time']}] {entry['action']}{details}")
        print("\nThank you for using GestureMed! 🏥")
        print("=" * 80)
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.hand_tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = GestureMedApp()
    app.run()
