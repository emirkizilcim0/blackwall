class NSLKDDDecoder:
    """
    Decoder for NSL-KDD numeric labels and features
    Based on NSL-KDD dataset documentation
    """
    
    # Label mapping based on NSL-KDD documentation
    LABEL_MAPPING = {
        'normal': 'normal',
        # DoS attacks
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 
        'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
        'processtable': 'dos', 'worm': 'dos',
        # Probe attacks
        'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        # R2L attacks
        'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
        'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l',
        'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
        'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
        # U2R attacks
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r',
        'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
    }
    
    # Numeric to string label mapping (you might need to adjust these based on your data)
    NUMERIC_LABEL_MAPPING = {
        1: 'normal',
        2: 'back', 3: 'buffer_overflow', 4: 'ftp_write', 5: 'guess_passwd',
        6: 'imap', 7: 'ipsweep', 8: 'land', 9: 'loadmodule', 10: 'multihop',
        11: 'neptune', 12: 'nmap', 13: 'perl', 14: 'phf', 15: 'pod',
        16: 'portsweep', 17: 'rootkit', 18: 'satan', 19: 'smurf', 20: 'spy',
        21: 'teardrop', 22: 'warezclient', 23: 'warezmaster'
        # Add more mappings as needed based on your dataset
    }
    
    # Protocol types
    PROTOCOL_MAPPING = {
        1: 'tcp',
        2: 'udp', 
        3: 'icmp'
    }
    
    # Service types (abbreviated - there are 70 services)
    SERVICE_MAPPING = {
        1: 'http', 2: 'smtp', 3: 'finger', 4: 'domain_u', 5: 'auth',
        6: 'telnet', 7: 'ftp', 8: 'eco_i', 9: 'ntp_u', 10: 'ecr_i',
        # Add more as needed
    }
    
    # Flag types
    FLAG_MAPPING = {
        1: 'SF', 2: 'S1', 3: 'REJ', 4: 'S2', 5: 'S0',
        6: 'S3', 7: 'RSTO', 8: 'RSTR', 9: 'RSTOS0', 10: 'OTH'
    }
    
    @classmethod
    def decode_label(cls, numeric_label):
        """Convert numeric label to string label"""
        return cls.NUMERIC_LABEL_MAPPING.get(numeric_label, f'unknown_{numeric_label}')
    
    @classmethod
    def decode_protocol(cls, numeric_protocol):
        """Convert numeric protocol to string"""
        return cls.PROTOCOL_MAPPING.get(numeric_protocol, f'unknown_{numeric_protocol}')
    
    @classmethod
    def categorize_attack(cls, label):
        """Categorize attack type"""
        if label == 'normal':
            return 'normal'
        return cls.LABEL_MAPPING.get(label, 'unknown')