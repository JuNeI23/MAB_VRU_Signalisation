"""
Unit tests for VRU simulation models.
"""
import pytest
from simulation.models import Node, User, Infrastructure, Message
from simulation.protocols import Protocole

def test_user_initialization():
    """Test User class initialization with valid parameters."""
    user = User(
        usager_id="test_user",
        x=10.0,
        y=20.0,
        angle=45.0,
        speed=30.0,
        position=100.0,
        lane="lane_0",
        time=0.0,
        usager_type="car",
        categorie="vehicule"
    )
    
    assert user.user_id == "test_user"
    assert user.x == 10.0
    assert user.y == 20.0
    assert user.angle == 45.0
    assert user.speed == 30.0
    assert user.position == 100.0
    assert user.lane == "lane_0"
    assert user.time == 0.0
    assert user.usager_type == "car"
    assert user.categorie == "vehicule"

def test_infrastructure_initialization():
    """Test Infrastructure class initialization."""
    protocol = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)
    infra = Infrastructure(
        id="infra_1",
        protocol=protocol,
        x=0.0,
        y=0.0,
        processing_capacity=100,
        time=0.0
    )
    
    assert infra.user_id == "infra_1"
    assert infra.x == 0.0
    assert infra.y == 0.0
    assert infra.processing_capacity == 100
    assert infra.time == 0.0
    assert infra.protocol == protocol
    assert infra.user_type == "Infrastructure"
    assert infra.categorie == "infrastructure"

def test_message_initialization():
    """Test Message class initialization and validation."""
    msg = Message(
        priority=1,
        sender_id="sender_1",
        receiver_id="receiver_1",
        size=100.0
    )
    
    assert msg.priority == 1
    assert msg.sender_id == "sender_1"
    assert msg.receiver_id == "receiver_1"
    assert msg.size == 100.0
    assert msg.timestamp > 0

def test_invalid_message():
    """Test Message validation for invalid parameters."""
    with pytest.raises(ValueError):
        Message(priority=1, sender_id="", receiver_id="receiver_1", size=100.0)
        
    with pytest.raises(ValueError):
        Message(priority=1, sender_id="sender_1", receiver_id="receiver_1", size=0)

def test_node_distance_calculation():
    """Test distance calculation between nodes."""
    node1 = User(
        usager_id="user1",
        x=0.0, y=0.0,
        angle=0.0, speed=0.0,
        position=0.0, lane="lane_0",
        time=0.0
    )
    
    node2 = User(
        usager_id="user2",
        x=3.0, y=4.0,
        angle=0.0, speed=0.0,
        position=0.0, lane="lane_0",
        time=0.0
    )
    
    # Distance should be 5.0 (3-4-5 triangle)
    assert abs(node1.distance_to(node2) - 5.0) < 1e-10

def test_node_range_check():
    """Test range checking between nodes."""
    node1 = User(
        usager_id="user1",
        x=0.0, y=0.0,
        angle=0.0, speed=0.0,
        position=0.0, lane="lane_0",
        time=0.0
    )
    
    node2 = User(
        usager_id="user2",
        x=80.0, y=0.0,
        angle=0.0, speed=0.0,
        position=0.0, lane="lane_0",
        time=0.0
    )
    
    # VRU range is 90.0, so node2 should be in range
    assert node1.within_range(node2)
    
    # Move node2 outside of range
    node2.x = 100.0
    assert not node1.within_range(node2)
