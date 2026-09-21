import random
from typing import Dict, List, Tuple
import torch


# Constants for Entity and Edge Types
NODE_TYPES = {
    0: "User",
    1: "Device",
    2: "Role",
    3: "Resource"
}

EDGE_TYPES = {
    0: "uses",             # User -> Device
    1: "has_role",         # User -> Role
    2: "requests_access",  # User -> Resource
    3: "accesses",         # Device -> Resource
    4: "permits"           # Role -> Resource
}


class CloudAccessGraph:
    """Represents a single Heterogeneous Cloud Access Activity Graph."""
    def __init__(
        self,
        node_names: List[str],
        node_types: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        label: int = 0,  # 0: Normal, 1: Anomaly
        scenario_name: str = "Normal Cloud Access",
        metadata: Dict = None
    ):
        self.node_names = node_names
        self.node_types = node_types  # [num_nodes]
        self.node_features = node_features  # [num_nodes, feature_dim]
        self.edge_index = edge_index  # [2, num_edges]
        self.edge_types = edge_types  # [num_edges]
        self.label = label
        self.scenario_name = scenario_name
        self.metadata = metadata or {}

    def to_dict(self):
        """Serializes graph structure for UI rendering and REST responses."""
        nodes = []
        for i, name in enumerate(self.node_names):
            t_id = self.node_types[i].item()
            nodes.append({
                "id": i,
                "name": name,
                "type": NODE_TYPES[t_id],
                "type_id": t_id,
                "features": [round(f, 3) for f in self.node_features[i].tolist()]
            })

        edges = []
        for e in range(self.edge_index.size(1)):
            src = self.edge_index[0, e].item()
            dst = self.edge_index[1, e].item()
            e_id = self.edge_types[e].item()
            edges.append({
                "source": src,
                "target": dst,
                "source_name": self.node_names[src],
                "target_name": self.node_names[dst],
                "relation": EDGE_TYPES[e_id],
                "relation_id": e_id
            })

        return {
            "scenario": self.scenario_name,
            "label": self.label,
            "is_anomaly": bool(self.label == 1),
            "metadata": self.metadata,
            "nodes": nodes,
            "edges": edges
        }


class CloudGraphGenerator:
    """
    Generates realistic Cloud IAM access graphs for both normal corporate operations
    and advanced threat vectors (credential compromise, privilege escalation, data exfiltration).
    """
    def __init__(self, feature_dim: int = 8):
        self.feature_dim = feature_dim

    def generate_normal_graph(self, dept: str = None) -> CloudAccessGraph:
        """Generates a legitimate corporate access graph (e.g. Engineering, DevOps, SecOps)."""
        depts = ["Engineering", "DevOps", "Finance", "SecOps", "Support"]
        chosen_dept = dept or random.choice(depts)

        # 1. User
        user_name = f"{chosen_dept}_Engineer_{random.randint(101, 999)}"
        # Features: [dept_id, seniority, mfa_active, location_risk, request_rate, prev_alerts, session_len, auth_protocol]
        user_feat = [
            depts.index(chosen_dept) / 5.0,
            random.uniform(0.5, 0.95),  # seniority
            1.0,                        # MFA active (compliant)
            0.05,                       # Low location risk (corporate IP/office)
            random.uniform(0.1, 0.4),   # Normal request rate
            0.0,                        # Zero previous security alerts
            random.uniform(0.2, 0.6),   # Normal session length
            0.9                         # Modern OAuth2/SAML protocol
        ]

        # 2. Device (Corporate Managed Laptop)
        device_name = f"Corp_{chosen_dept}_MacBook_{random.randint(1, 50)}"
        # Features: [device_trust_score, is_managed, os_type, compliance_score, disk_encrypted, edr_active, ip_reputation, vpn_used]
        device_feat = [
            random.uniform(0.85, 0.99), # High trust score
            1.0,                        # Managed device
            0.8,                        # macOS/Enterprise Linux
            random.uniform(0.9, 1.0),   # 100% compliance
            1.0,                        # Full disk encryption
            1.0,                        # CrowdStrike / EDR active
            0.95,                       # Clean IP reputation
            1.0                         # Corporate VPN verified
        ]

        # 3. Role
        role_name = f"{chosen_dept}_Standard_Role"
        # Features: [privilege_level, is_admin, max_session_hrs, scope_breadth, mfa_required, cross_account, policy_actions, access_tier]
        role_feat = [
            0.35 if chosen_dept != "SecOps" else 0.7,  # Moderate privilege
            0.0,                        # Not root admin
            0.33,                       # 8-hour session limit
            0.3,                        # Scoped breadth
            1.0,                        # MFA required
            0.0,                        # Single account scope
            0.4,                        # Read/Write project scope
            0.5                         # Internal tier
        ]

        # 4. Resource
        res_pool = {
            "Engineering": ("App_Git_Repo_Prod", 0.4, 0.0),
            "DevOps": ("K8s_Cluster_Staging", 0.5, 0.0),
            "Finance": ("Invoicing_DB_Read", 0.6, 0.0),
            "SecOps": ("CloudTrail_Audit_Bucket", 0.7, 0.0),
            "Support": ("Zendesk_CRM_Service", 0.3, 0.0)
        }
        res_info = res_pool.get(chosen_dept, ("App_Git_Repo_Prod", 0.4, 0.0))
        res_name = res_info[0]
        # Features: [sensitivity_tier, is_production, resource_type, encryption_enabled, is_public, pii_contained, region_id, audit_logging]
        res_feat = [
            res_info[1],                # Sensitivity tier
            res_info[2],                # Non-critical prod flag
            0.5,                        # Microservice / S3 / DB
            1.0,                        # AES-256 encrypted
            0.0,                        # Zero public access
            0.0,                        # No raw PII
            0.2,                        # Primary cloud region
            1.0                         # CloudTrail audit active
        ]

        # Assemble graph components
        node_names = [user_name, device_name, role_name, res_name]
        node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

        # Edges (Standard normal IAM workflow):
        # 0: User(0) -uses-> Device(1)
        # 1: User(0) -has_role-> Role(2)
        # 2: User(0) -requests_access-> Resource(3)
        # 3: Device(1) -accesses-> Resource(3)
        # 4: Role(2) -permits-> Resource(3)
        edge_index = torch.tensor([
            [0, 0, 0, 1, 2],
            [1, 2, 3, 3, 3]
        ], dtype=torch.long)
        edge_types = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        return CloudAccessGraph(
            node_names=node_names,
            node_types=node_types,
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            label=0,
            scenario_name=f"Legitimate {chosen_dept} Routine Access",
            metadata={"department": chosen_dept, "risk_level": "LOW", "mfa_verified": True}
        )

    def generate_attack_graph(self, attack_type: int = 1) -> CloudAccessGraph:
        """
        Generates 5 distinct Cloud Threat Vectors:
        1: Stolen Credential via Unknown BYOD Device
        2: Privilege Escalation Attack to Production Database
        3: Rogue Lateral Movement & Data Exfiltration on S3
        4: Off-Hours Cross-Department Breach
        5: Compromised Root Role with Disabled MFA
        """
        if attack_type == 1:
            # 1. Stolen Credential via Unknown BYOD Device
            user_name = "Dev_User_Compromised"
            device_name = "Rogue_Android_Unknown_BYOD"
            role_name = "AWS_Global_Administrator"
            res_name = "Prod_Customer_Payments_DB"

            user_feat = [0.2, 0.4, 0.0, 0.95, 0.9, 0.8, 0.1, 0.2]  # No MFA, extreme location risk, spike rate
            device_feat = [0.15, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0] # Untrusted, unmanaged, no EDR, dirty IP
            role_feat = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]   # Root admin tier
            res_feat = [1.0, 1.0, 0.9, 1.0, 0.0, 1.0, 0.2, 1.0]    # Sensitive financial DB

            node_names = [user_name, device_name, role_name, res_name]
            node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

            # Malicious direct access bypassing role permission check
            edge_index = torch.tensor([
                [0, 0, 1, 1],
                [1, 3, 3, 2]
            ], dtype=torch.long)
            edge_types = torch.tensor([0, 2, 3, 1], dtype=torch.long)

            return CloudAccessGraph(
                node_names=node_names,
                node_types=node_types,
                node_features=node_features,
                edge_index=edge_index,
                edge_types=edge_types,
                label=1,
                scenario_name="Credential Theft via Unmanaged BYOD Device",
                metadata={"threat_type": "Identity Compromise", "risk_level": "CRITICAL", "details": "Unmanaged rogue device attempting direct DB access with hijacked session"}
            )

        elif attack_type == 2:
            # 2. Privilege Escalation Attack
            user_name = "Junior_Support_Analyst"
            device_name = "Corp_Support_Workstation"
            role_name = "SuperAdmin_KMS_Master_Role"
            res_name = "KMS_Master_Encryption_Key"

            user_feat = [0.8, 0.1, 1.0, 0.2, 0.85, 0.6, 0.9, 0.5]  # Low seniority, sudden privilege spike
            device_feat = [0.75, 1.0, 0.5, 0.85, 1.0, 1.0, 0.8, 1.0] # Normal machine
            role_feat = [1.0, 1.0, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0]  # SuperAdmin Role
            res_feat = [1.0, 1.0, 0.95, 1.0, 0.0, 1.0, 0.2, 1.0]   # KMS Root Key

            node_names = [user_name, device_name, role_name, res_name]
            node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

            edge_index = torch.tensor([
                [0, 0, 0, 2],
                [1, 2, 3, 3]
            ], dtype=torch.long)
            edge_types = torch.tensor([0, 1, 2, 4], dtype=torch.long)

            return CloudAccessGraph(
                node_names=node_names,
                node_types=node_types,
                node_features=node_features,
                edge_index=edge_index,
                edge_types=edge_types,
                label=1,
                scenario_name="Unauthorized IAM Privilege Escalation",
                metadata={"threat_type": "Privilege Escalation", "risk_level": "HIGH", "details": "Junior tier user attempting to assume high-privilege KMS Master Role"}
            )

        elif attack_type == 3:
            # 3. Rogue S3 Data Exfiltration
            user_name = "Compromised_Worker_Service"
            device_name = "EC2_Automated_Runner"
            role_name = "Restricted_ReadOnly_Role"
            res_name = "S3_PII_Customer_Vault_Prod"

            user_feat = [0.1, 0.2, 0.0, 0.8, 0.99, 0.9, 0.95, 0.3]
            device_feat = [0.4, 1.0, 0.9, 0.5, 0.0, 0.0, 0.3, 0.0]
            role_feat = [0.2, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.2]  # Underprivileged role
            res_feat = [1.0, 1.0, 0.8, 1.0, 0.0, 1.0, 0.5, 1.0]   # High sensitivity S3

            node_names = [user_name, device_name, role_name, res_name]
            node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

            # Direct exfiltration bypass
            edge_index = torch.tensor([
                [0, 1, 0],
                [1, 3, 3]
            ], dtype=torch.long)
            edge_types = torch.tensor([0, 3, 2], dtype=torch.long)

            return CloudAccessGraph(
                node_names=node_names,
                node_types=node_types,
                node_features=node_features,
                edge_index=edge_index,
                edge_types=edge_types,
                label=1,
                scenario_name="Rogue S3 Bulk Data Exfiltration",
                metadata={"threat_type": "Data Exfiltration", "risk_level": "CRITICAL", "details": "High-volume read request to restricted PII bucket bypassing role policy"}
            )

        elif attack_type == 4:
            # 4. Off-Hours Cross-Department Breach
            user_name = "Marketing_Contractor_01"
            device_name = "Personal_Unverified_Laptop"
            role_name = "Finance_Billing_Ledger_Role"
            res_name = "Quarterly_Financial_Tax_Records"

            user_feat = [0.9, 0.1, 0.0, 0.88, 0.7, 0.5, 0.1, 0.2]
            device_feat = [0.3, 0.0, 0.4, 0.3, 0.0, 0.0, 0.4, 0.0]
            role_feat = [0.75, 0.0, 0.4, 0.5, 1.0, 0.0, 0.6, 0.7]
            res_feat = [0.9, 1.0, 0.7, 1.0, 0.0, 0.8, 0.1, 1.0]

            node_names = [user_name, device_name, role_name, res_name]
            node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

            edge_index = torch.tensor([
                [0, 0, 1, 2],
                [1, 2, 3, 3]
            ], dtype=torch.long)
            edge_types = torch.tensor([0, 1, 3, 4], dtype=torch.long)

            return CloudAccessGraph(
                node_names=node_names,
                node_types=node_types,
                node_features=node_features,
                edge_index=edge_index,
                edge_types=edge_types,
                label=1,
                scenario_name="Off-Hours Cross-Department Resource Breach",
                metadata={"threat_type": "Cross-Domain Violation", "risk_level": "HIGH", "details": "Marketing identity attempting access to confidential financial tax ledger off-hours"}
            )

        else:
            # 5. Compromised Root Role with Disabled MFA
            user_name = "External_Script_Attacker"
            device_name = "Tor_Exit_Node_Virtual_Client"
            role_name = "Cloud_Root_Admin"
            res_name = "IAM_Policy_Management_Service"

            user_feat = [0.0, 0.99, 0.0, 0.99, 0.99, 1.0, 0.05, 0.1]
            device_feat = [0.05, 0.0, 0.1, 0.0, 0.0, 0.0, 0.05, 0.0]
            role_feat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            res_feat = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.1, 1.0]

            node_names = [user_name, device_name, role_name, res_name]
            node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

            edge_index = torch.tensor([
                [0, 0, 1, 2],
                [1, 2, 3, 3]
            ], dtype=torch.long)
            edge_types = torch.tensor([0, 1, 3, 4], dtype=torch.long)

            return CloudAccessGraph(
                node_names=node_names,
                node_types=node_types,
                node_features=node_features,
                edge_index=edge_index,
                edge_types=edge_types,
                label=1,
                scenario_name="Root IAM Hijacking via Untrusted Network",
                metadata={"threat_type": "Account Takeover", "risk_level": "CRITICAL", "details": "Anomalous root policy modification request originating from an anonymous proxy network"}
            )

    def generate_subtle_anomaly(self) -> CloudAccessGraph:
        """
        Generates borderline anomaly graphs that look structurally similar to normal
        but have subtle relational inconsistencies. These are harder to detect and
        require hierarchical source-dest + edge type modeling to catch.
        """
        variant = random.randint(1, 4)

        if variant == 1:
            # Slightly elevated privileges - normal-looking features but wrong role-resource pair
            user_name = f"Contractor_{random.randint(100,999)}"
            device_name = f"Corp_Laptop_{random.randint(1,50)}"
            role_name = "Finance_Admin_Role"
            res_name = "Engineering_Source_Code_Repo"

            user_feat = [0.6, 0.45, 1.0, 0.15, random.uniform(0.2, 0.5), 0.1, 0.4, 0.85]
            device_feat = [random.uniform(0.7, 0.9), 1.0, 0.8, 0.88, 1.0, 1.0, 0.85, 1.0]
            role_feat = [0.6, 0.0, 0.4, 0.55, 1.0, 0.0, 0.5, 0.6]
            res_feat = [0.5, 0.0, 0.6, 1.0, 0.0, 0.0, 0.2, 1.0]
        elif variant == 2:
            # Normal user but accessing via unusual edge path (device->resource direct without role)
            user_name = f"Engineer_{random.randint(100,999)}"
            device_name = f"Shared_Terminal_{random.randint(1,10)}"
            role_name = "Dev_ReadWrite_Role"
            res_name = "Staging_Database"

            user_feat = [0.1, 0.65, 1.0, 0.2, random.uniform(0.1, 0.4), 0.0, 0.5, 0.9]
            device_feat = [random.uniform(0.5, 0.75), 1.0, 0.7, 0.78, 1.0, 0.8, 0.7, 1.0]
            role_feat = [0.35, 0.0, 0.33, 0.3, 1.0, 0.0, 0.4, 0.5]
            res_feat = [0.55, 0.0, 0.5, 1.0, 0.0, 0.0, 0.2, 1.0]
        elif variant == 3:
            # Normal-looking but with cross-department role assumption
            user_name = f"Marketing_Analyst_{random.randint(100,999)}"
            device_name = f"Corp_MacBook_{random.randint(1,50)}"
            role_name = "DevOps_Deploy_Role"
            res_name = "CI_CD_Pipeline_Service"

            user_feat = [0.8, 0.5, 1.0, 0.1, random.uniform(0.2, 0.4), 0.15, 0.5, 0.9]
            device_feat = [random.uniform(0.8, 0.95), 1.0, 0.8, 0.92, 1.0, 1.0, 0.9, 1.0]
            role_feat = [0.55, 0.0, 0.5, 0.6, 1.0, 0.0, 0.65, 0.6]
            res_feat = [0.6, 1.0, 0.7, 1.0, 0.0, 0.0, 0.2, 1.0]
        else:
            # Slightly anomalous session timing/rate but otherwise standard
            user_name = f"Support_Agent_{random.randint(100,999)}"
            device_name = f"Corp_Desktop_{random.randint(1,30)}"
            role_name = "Support_CRM_Role"
            res_name = "Customer_Records_DB"

            user_feat = [0.7, 0.3, 1.0, 0.25, random.uniform(0.7, 0.95), 0.3, 0.85, 0.8]
            device_feat = [random.uniform(0.6, 0.85), 1.0, 0.5, 0.8, 1.0, 1.0, 0.75, 1.0]
            role_feat = [0.4, 0.0, 0.3, 0.35, 1.0, 0.0, 0.35, 0.4]
            res_feat = [0.7, 1.0, 0.6, 1.0, 0.0, 0.5, 0.2, 1.0]

        node_names = [user_name, device_name, role_name, res_name]
        node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        node_features = torch.tensor([user_feat, device_feat, role_feat, res_feat], dtype=torch.float)

        # Structurally identical to normal but semantically wrong relations
        edge_index = torch.tensor([
            [0, 0, 0, 1, 2],
            [1, 2, 3, 3, 3]
        ], dtype=torch.long)
        edge_types = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        return CloudAccessGraph(
            node_names=node_names,
            node_types=node_types,
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            label=1,
            scenario_name=f"Subtle Cross-Domain Access (Variant {variant})",
            metadata={"threat_type": "Subtle Anomaly", "risk_level": "MEDIUM"}
        )

    def create_dataset(self, num_normal: int = 500, num_attack: int = 150) -> Tuple[List[CloudAccessGraph], List[CloudAccessGraph]]:
        """
        Generates train (all normal) and test (mixed normal, overt attacks, and subtle anomalies).
        Subtle anomalies ensure ablation variants show differentiated detection capability.
        """
        train_graphs = [self.generate_normal_graph() for _ in range(num_normal)]

        # Add feature noise to some training graphs (so model doesn't overfit to exact patterns)
        for g in train_graphs:
            noise = torch.randn_like(g.node_features) * 0.05
            g.node_features = torch.clamp(g.node_features + noise, 0.0, 1.0)

        test_normal = [self.generate_normal_graph() for _ in range(num_attack)]
        # Mix of overt attacks (easy) and subtle anomalies (hard - require hierarchical relations)
        num_overt = num_attack // 2
        num_subtle = num_attack - num_overt
        test_overt = [self.generate_attack_graph(attack_type=(i % 5) + 1) for i in range(num_overt)]
        test_subtle = [self.generate_subtle_anomaly() for _ in range(num_subtle)]

        test_graphs = test_normal + test_overt + test_subtle
        random.shuffle(test_graphs)
        return train_graphs, test_graphs

