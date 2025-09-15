// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FL_SettlementChannel
 * @author Gemini
 * @notice A multi-party state channel to settle payments for a Cross-Silo
 * Federated Learning incentive mechanism. This contract acts as a trusted
 * escrow and settlement layer.
 */
contract FL_SettlementChannel {

    // --- State Variables ---

    enum ChannelState { Open, Closed }
    ChannelState public currentState;

    uint256 public constant DURATION = 24 hours;
    uint256 public immutable closeDeadline;

    address[] public participants;
    // Mapping from participant address to their initial net balance (positive = receive, negative = pay)
    mapping(address => int256) public netBalances;
    // Mapping to track funds deposited by payers
    mapping(address => uint256) public depositedFunds;

    uint256 public totalRequiredFromPayers;
    uint256 public nonce;

    // --- Events ---

    event ChannelOpened(address[] participants, int256[] balances);
    event FundsDeposited(address indexed payer, uint256 amount);
    event ChannelClosed(uint256 finalNonce);


    // --- Functions ---

    /**
     * @notice Deploys the contract, opening the channel.
     * @param _participants An array of all participant addresses.
     * @param _netBalances An array of the corresponding net balances.
     */
    constructor(address[] memory _participants, int256[] memory _netBalances) {
        require(_participants.length == _netBalances.length, "Input arrays must have same length");
        
        participants = _participants;
        int256 balanceSum = 0;
        uint256 requiredDeposit = 0;

        for (uint i = 0; i < _participants.length; i++) {
            netBalances[_participants[i]] = _netBalances[i];
            balanceSum += _netBalances[i];
            if (_netBalances[i] < 0) {
                requiredDeposit += uint256(-_netBalances[i]);
            }
        }

        require(balanceSum == 0, "Net balances must sum to zero");
        totalRequiredFromPayers = requiredDeposit;
        
        currentState = ChannelState.Open;
        closeDeadline = block.timestamp + DURATION;
        nonce = 0;

        emit ChannelOpened(_participants, _netBalances);
    }

    /**
     * @notice Allows payers to deposit their owed funds into the contract.
     */
    function deposit() external payable {
        require(currentState == ChannelState.Open, "Channel is not open");
        int256 balance = netBalances[msg.sender];
        require(balance < 0, "Only payers can deposit");

        uint256 requiredAmount = uint256(-balance);
        require(msg.value == requiredAmount, "Must deposit exact owed amount");
        require(depositedFunds[msg.sender] == 0, "Already deposited");

        depositedFunds[msg.sender] = msg.value;
        emit FundsDeposited(msg.sender, msg.value);
    }

    /**
     * @notice Closes the channel by providing the final state and all signatures.
     * @param finalBalances The final balances of all participants (should be 0 for all).
     * @param signatures An array of signatures, one from each participant in order.
     */
    function closeChannel(int256[] memory finalBalances, bytes[] memory signatures) external {
        require(currentState == ChannelState.Open, "Channel is not open");
        require(block.timestamp <= closeDeadline, "Deadline has passed");
        require(address(this).balance >= totalRequiredFromPayers, "All funds not deposited");
        require(signatures.length == participants.length, "Incorrect number of signatures");

        uint256 finalNonce = nonce + 1;
        
        // 1. Get the hash of the raw data
        bytes32 dataHash = getDataHash(finalBalances, finalNonce);
        // 2. Add the standard EIP-191 prefix to match the Python library's behavior
        bytes32 prefixedMessageHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", dataHash));
        
        for (uint i = 0; i < participants.length; i++) {
            // 3. Recover the signer from the prefixed hash
            address signer = recoverSigner(prefixedMessageHash, signatures[i]);
            require(signer == participants[i], "Invalid signature");
        }

        currentState = ChannelState.Closed;
        nonce = finalNonce;

        for (uint i = 0; i < participants.length; i++) {
            address participant = participants[i];
            int256 balance = netBalances[participant];
            if (balance > 0) {
                payable(participant).transfer(uint256(balance));
            }
        }
        
        emit ChannelClosed(finalNonce);
    }

    /**
     * @notice Hashes the raw channel state data. The EIP-191 prefix is applied by the client.
     */
    function getDataHash(int256[] memory _balances, uint256 _nonce) public view returns (bytes32) {
        return keccak256(abi.encode(address(this), _balances, _nonce));
    }

    /**
     * @dev Recovers the address of a signer from a signature.
     */
    function recoverSigner(bytes32 _messageHash, bytes memory _signature) internal pure returns (address) {
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(_signature);
        return ecrecover(_messageHash, v, r, s);
    }

    function splitSignature(bytes memory sig) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "invalid signature length");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
    }
}