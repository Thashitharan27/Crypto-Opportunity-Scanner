"""Researcher-facing rule-based strategy builder.

Users choose where the strategy may trade and then express every entry-affecting
condition through scoped Entry/Veto rules. DI pressure and support/resistance are
causal research evidence, not separate hidden/global filters.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from crypto_strategy_lab.strategy_profiles import RULE_INDICATORS
from crypto_strategy_lab.strategy_rule_model import (
    DIRECTION_MODES,
    MARKET_PERMISSIONS,
    REGIMES,
    SIDES,
    decompile_rules,
    infer_direction_mode,
    infer_market_permissions,
    is_categorical_evidence,
    new_rule,
    normalize_rule,
    rule_operator_options,
    rule_value_options,
)


EVIDENCE_LABELS = {
    "DI_SPREAD": "DI Spread",
    "DIRECTIONAL_DI": "Directional DI",
    "DI_PRESSURE_STATE": "DI Pressure State",
    "DI_SPREAD_CHANGE": "DI Spread Change",
    "DIRECTIONAL_DI_CHANGE": "Directional DI Change",
    "OPPOSING_DI_CHANGE": "Opposing DI Change",
    "ADX": "ADX",
    "ADX_CHANGE": "ADX Change (1 bar)",
    "ATR_PCT": "ATR % (decimal)",
    "RSI": "RSI",
    "BB_WIDTH": "BB Width (decimal)",
    "CLOSE_LOCATION": "Close Location",
    "MOMENTUM": "Momentum Return",
    "VWAP_DISTANCE": "VWAP Distance (ATR)",
    "SR_NEAR_SUPPORT": "S/R — Near Support",
    "SR_NEAR_RESISTANCE": "S/R — Near Resistance",
    "SR_INSIDE_SUPPORT_ZONE": "S/R — Inside Support Zone",
    "SR_INSIDE_RESISTANCE_ZONE": "S/R — Inside Resistance Zone",
    "SR_SUPPORT_STATE": "S/R — Support State",
    "SR_RESISTANCE_STATE": "S/R — Resistance State",
    "SR_SUPPORT_HELD": "S/R — Support Held",
    "SR_RESISTANCE_HELD": "S/R — Resistance Held",
    "SR_TRADE_LOCATION_RATING": "S/R — Trade Location Rating",
    "SR_ROOM_IN_DIRECTION_ATR": "S/R — Room In Direction (ATR)",
    "SR_SUPPORT_DISTANCE_ATR": "S/R — Support Distance (ATR)",
    "SR_RESISTANCE_DISTANCE_ATR": "S/R — Resistance Distance (ATR)",
    "SR_SUPPORT_REJECTION_ATR": "S/R — Support Rejection (ATR)",
    "SR_RESISTANCE_REJECTION_ATR": "S/R — Resistance Rejection (ATR)",
    "OI_CHANGE_PCT_5M": "OI Change 5m (decimal)",
    "OI_CHANGE_PCT_1H": "OI Change 1h (decimal)",
    "OI_CHANGE_PCT_24H": "OI Change 24h (decimal)",
    "OI_ZSCORE_7D": "OI Z-Score (7d)",
    "PRICE_CHANGE_PCT_1H": "Price Change 1h (decimal)",
    "OI_VS_PRICE_STATE_1H": "Price / OI State (1h)",
    "TOP_TRADER_ACCOUNT_BIAS": "Top Trader Account Bias (ratio − 1)",
    "TOP_TRADER_POSITION_BIAS": "Top Trader Position Bias (ratio − 1)",
    "GLOBAL_LONG_SHORT_ACCOUNT_BIAS": "Global Long/Short Account Bias (ratio − 1)",
    "TAKER_LONG_SHORT_VOLUME_BIAS": "Taker Long/Short Volume Bias (ratio − 1)",
    "FUNDING_RATE_BPS": "Funding Rate (bps)",
    "FUNDING_BIAS": "Funding Bias",
    "FUNDING_24H_SUM_BPS": "Funding 24h Sum (bps)",
    "FUNDING_CHANGE_BPS": "Funding Change (bps)",
    "FUNDING_3_EVENT_MEAN_BPS": "Funding 3-Event Mean (bps)",
    "FUNDING_ZSCORE_7D": "Funding Z-Score (7d)",
    "FUNDING_EXTREME_POSITIVE": "Funding Extreme Positive",
    "FUNDING_EXTREME_NEGATIVE": "Funding Extreme Negative",
    "MARK_INDEX_BASIS_BPS": "Mark − Index Basis (bps)",
    "MARK_INDEX_BASIS_STATE": "Mark / Index Basis State",
    "MARK_INDEX_BASIS_ZSCORE_7D": "Mark / Index Basis Z-Score (7d)",
    "TRADE_MARK_BASIS_BPS": "Trade − Mark Basis (bps)",
    "TRADE_INDEX_BASIS_BPS": "Trade − Index Basis (bps)",
    "PREMIUM_INDEX_ZSCORE_7D": "Premium Index Z-Score (7d)",
    "TAKER_BUY_SELL_RATIO": "Taker Buy / Sell Ratio",
    "TAKER_DELTA_PCT": "Taker Delta (source interval, decimal)",
    "TAKER_DELTA_PCT_15M": "Taker Delta 15m (decimal)",
    "TAKER_DELTA_PCT_1H": "Taker Delta 1h (decimal)",
    "TAKER_FLOW_PERSISTENCE": "Taker Flow Persistence (0–1)",
}
EVIDENCE_GROUPS = (
    (
        "Directional / DI",
        (
            "DIRECTIONAL_DI",
            "DI_SPREAD",
            "DI_PRESSURE_STATE",
            "DIRECTIONAL_DI_CHANGE",
            "OPPOSING_DI_CHANGE",
            "DI_SPREAD_CHANGE",
        ),
    ),
    ("Trend & Volatility", ("ADX", "ADX_CHANGE", "ATR_PCT", "BB_WIDTH")),
    (
        "Momentum & Price",
        ("RSI", "MOMENTUM", "CLOSE_LOCATION", "VWAP_DISTANCE"),
    ),
    (
        "Futures — Open Interest & Positioning",
        (
            "OI_VS_PRICE_STATE_1H",
            "OI_CHANGE_PCT_5M",
            "OI_CHANGE_PCT_1H",
            "OI_CHANGE_PCT_24H",
            "OI_ZSCORE_7D",
            "PRICE_CHANGE_PCT_1H",
            "TOP_TRADER_ACCOUNT_BIAS",
            "TOP_TRADER_POSITION_BIAS",
            "GLOBAL_LONG_SHORT_ACCOUNT_BIAS",
            "TAKER_LONG_SHORT_VOLUME_BIAS",
        ),
    ),
    (
        "Futures — Funding",
        (
            "FUNDING_BIAS",
            "FUNDING_RATE_BPS",
            "FUNDING_24H_SUM_BPS",
            "FUNDING_CHANGE_BPS",
            "FUNDING_3_EVENT_MEAN_BPS",
            "FUNDING_ZSCORE_7D",
            "FUNDING_EXTREME_POSITIVE",
            "FUNDING_EXTREME_NEGATIVE",
        ),
    ),
    (
        "Futures — Basis / Premium",
        (
            "MARK_INDEX_BASIS_STATE",
            "MARK_INDEX_BASIS_BPS",
            "MARK_INDEX_BASIS_ZSCORE_7D",
            "TRADE_MARK_BASIS_BPS",
            "TRADE_INDEX_BASIS_BPS",
            "PREMIUM_INDEX_ZSCORE_7D",
        ),
    ),
    (
        "Futures — Taker Flow",
        (
            "TAKER_BUY_SELL_RATIO",
            "TAKER_DELTA_PCT",
            "TAKER_DELTA_PCT_15M",
            "TAKER_DELTA_PCT_1H",
            "TAKER_FLOW_PERSISTENCE",
        ),
    ),
    (
        "Support & Resistance",
        (
            "SR_TRADE_LOCATION_RATING",
            "SR_ROOM_IN_DIRECTION_ATR",
            "SR_NEAR_SUPPORT",
            "SR_NEAR_RESISTANCE",
            "SR_INSIDE_SUPPORT_ZONE",
            "SR_INSIDE_RESISTANCE_ZONE",
        ),
    ),
    (
        "Support & Resistance — Advanced",
        (
            "SR_SUPPORT_STATE",
            "SR_RESISTANCE_STATE",
            "SR_SUPPORT_HELD",
            "SR_RESISTANCE_HELD",
            "SR_SUPPORT_DISTANCE_ATR",
            "SR_RESISTANCE_DISTANCE_ATR",
            "SR_SUPPORT_REJECTION_ATR",
            "SR_RESISTANCE_REJECTION_ATR",
        ),
    ),
)

# Category-first menu structure. Leaves are evidence IDs; nested tuples are
# (submenu label, children). Keeping EVIDENCE_GROUPS above preserves the flat
# research grouping used elsewhere while the popup remains compact.
EVIDENCE_MENU_TREE = (
    (
        "Directional / DI",
        (
            "DIRECTIONAL_DI",
            "DI_SPREAD",
            "DI_PRESSURE_STATE",
            "DIRECTIONAL_DI_CHANGE",
            "OPPOSING_DI_CHANGE",
            "DI_SPREAD_CHANGE",
        ),
    ),
    ("Trend & Volatility", ("ADX", "ADX_CHANGE", "ATR_PCT", "BB_WIDTH")),
    (
        "Momentum & Price",
        ("RSI", "MOMENTUM", "CLOSE_LOCATION", "VWAP_DISTANCE"),
    ),
    (
        "Futures",
        (
            (
                "Open Interest & Positioning",
                (
                    "OI_VS_PRICE_STATE_1H",
                    "OI_CHANGE_PCT_5M",
                    "OI_CHANGE_PCT_1H",
                    "OI_CHANGE_PCT_24H",
                    "OI_ZSCORE_7D",
                    "PRICE_CHANGE_PCT_1H",
                    "TOP_TRADER_ACCOUNT_BIAS",
                    "TOP_TRADER_POSITION_BIAS",
                    "GLOBAL_LONG_SHORT_ACCOUNT_BIAS",
                    "TAKER_LONG_SHORT_VOLUME_BIAS",
                ),
            ),
            (
                "Funding",
                (
                    "FUNDING_BIAS",
                    "FUNDING_RATE_BPS",
                    "FUNDING_24H_SUM_BPS",
                    "FUNDING_CHANGE_BPS",
                    "FUNDING_3_EVENT_MEAN_BPS",
                    "FUNDING_ZSCORE_7D",
                    "FUNDING_EXTREME_POSITIVE",
                    "FUNDING_EXTREME_NEGATIVE",
                ),
            ),
            (
                "Basis / Premium",
                (
                    "MARK_INDEX_BASIS_STATE",
                    "MARK_INDEX_BASIS_BPS",
                    "MARK_INDEX_BASIS_ZSCORE_7D",
                    "TRADE_MARK_BASIS_BPS",
                    "TRADE_INDEX_BASIS_BPS",
                    "PREMIUM_INDEX_ZSCORE_7D",
                ),
            ),
            (
                "Taker Flow",
                (
                    "TAKER_BUY_SELL_RATIO",
                    "TAKER_DELTA_PCT",
                    "TAKER_DELTA_PCT_15M",
                    "TAKER_DELTA_PCT_1H",
                    "TAKER_FLOW_PERSISTENCE",
                ),
            ),
        ),
    ),
    (
        "Support & Resistance",
        (
            "SR_TRADE_LOCATION_RATING",
            "SR_ROOM_IN_DIRECTION_ATR",
            "SR_NEAR_SUPPORT",
            "SR_NEAR_RESISTANCE",
            "SR_INSIDE_SUPPORT_ZONE",
            "SR_INSIDE_RESISTANCE_ZONE",
            (
                "Advanced S/R",
                (
                    "SR_SUPPORT_STATE",
                    "SR_RESISTANCE_STATE",
                    "SR_SUPPORT_HELD",
                    "SR_RESISTANCE_HELD",
                    "SR_SUPPORT_DISTANCE_ATR",
                    "SR_RESISTANCE_DISTANCE_ATR",
                    "SR_SUPPORT_REJECTION_ATR",
                    "SR_RESISTANCE_REJECTION_ATR",
                ),
            ),
        ),
    ),
)

OPERATOR_LABELS = {
    "GT": ">",
    "GTE": "≥",
    "LT": "<",
    "LTE": "≤",
    "BETWEEN": "Between",
    "OUTSIDE": "Outside range",
    "IS": "Is",
    "IS_NOT": "Is Not",
}
DIRECTION_LABELS = {"DI": "DI Direction", "DMI_TREND": "DMI Trend — Baseline"}


def _humanize(value: str) -> str:
    return str(value).replace("_", " ").title()


def _evidence_menu_paths() -> dict[str, tuple[str, ...]]:
    """Return searchable category paths for every evidence leaf in the menu tree."""
    result: dict[str, tuple[str, ...]] = {}

    def visit(items, path):
        for item in items:
            if isinstance(item, str):
                result[item] = path
            else:
                label, children = item
                visit(children, (*path, label))

    for label, children in EVIDENCE_MENU_TREE:
        visit(children, (label,))
    return result


class EvidenceComboBox(QComboBox):
    """Evidence selector with compact categories plus global search."""

    def __init__(self, current: str | None = None, parent=None):
        super().__init__(parent)
        for evidence in RULE_INDICATORS:
            self.addItem(EVIDENCE_LABELS.get(evidence, evidence), evidence)
        index = self.findData(current)
        self.setCurrentIndex(max(index, 0))
        self.setToolTip("Browse evidence by category or search by name")

    def showPopup(self) -> None:
        menu = QMenu(self)
        menu.setMinimumWidth(max(self.width(), 360))

        search = QLineEdit(menu)
        search.setPlaceholderText("Search evidence…")
        search_action = QWidgetAction(menu)
        search_action.setDefaultWidget(search)
        menu.addAction(search_action)

        current = self.currentData()
        paths = _evidence_menu_paths()

        search_section = menu.addSection("Search Results")
        search_section.setVisible(False)
        search_actions = []
        for evidence in RULE_INDICATORS:
            label = EVIDENCE_LABELS.get(evidence, evidence)
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(evidence == current)
            action.setVisible(False)
            action.triggered.connect(
                lambda _checked=False, value=evidence: self._select_evidence(value)
            )
            path = paths.get(evidence, ("Other",))
            search_text = (
                f"{' '.join(path)} {label} {evidence.replace('_', ' ')}".casefold()
            )
            search_actions.append((action, search_text))

        no_matches = menu.addAction("No matching evidence")
        no_matches.setEnabled(False)
        no_matches.setVisible(False)

        category_separator = menu.addSeparator()
        category_actions = []
        tree_ids = set(paths)

        def add_items(parent_menu, items):
            for item in items:
                if isinstance(item, str):
                    if item not in RULE_INDICATORS:
                        continue
                    label = EVIDENCE_LABELS.get(item, item)
                    action = parent_menu.addAction(label)
                    action.setCheckable(True)
                    action.setChecked(item == current)
                    action.triggered.connect(
                        lambda _checked=False, value=item: self._select_evidence(value)
                    )
                else:
                    label, children = item
                    submenu = parent_menu.addMenu(label)
                    add_items(submenu, children)

        for label, children in EVIDENCE_MENU_TREE:
            submenu = menu.addMenu(label)
            add_items(submenu, children)
            category_actions.append(submenu.menuAction())

        uncategorized = tuple(
            evidence for evidence in RULE_INDICATORS if evidence not in tree_ids
        )
        if uncategorized:
            submenu = menu.addMenu("Other")
            add_items(submenu, uncategorized)
            category_actions.append(submenu.menuAction())

        def apply_filter(text: str) -> None:
            needle = text.strip().casefold()
            searching = bool(needle)
            any_match = False
            for action, search_text in search_actions:
                visible = searching and needle in search_text
                action.setVisible(visible)
                any_match = any_match or visible
            search_section.setVisible(searching and any_match)
            no_matches.setVisible(searching and not any_match)
            category_separator.setVisible(not searching)
            for action in category_actions:
                action.setVisible(not searching)

        search.textChanged.connect(apply_filter)
        search.setFocus()
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _select_evidence(self, evidence: str) -> None:
        index = self.findData(evidence)
        if index >= 0:
            self.setCurrentIndex(index)


class RuleTable(QTableWidget):
    """Compact scoped rule editor supporting numeric and categorical evidence."""

    changed = Signal()
    COLUMNS = ("evidence", "operator", "value", "value2", "regime", "side")

    def __init__(self, kind: str, parent=None):
        super().__init__(0, len(self.COLUMNS), parent)
        self.kind = kind
        self._ids: list[str] = []
        self.setHorizontalHeaderLabels(
            ("Evidence", "Condition", "Value", "Upper", "Market", "Side")
        )
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setMinimumHeight(150)
        self.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _combo(options, current):
        box = QComboBox()
        for native, label in options:
            box.addItem(label, native)
        index = box.findData(current)
        box.setCurrentIndex(max(index, 0))
        return box

    @staticmethod
    def _number(value: float):
        box = QDoubleSpinBox()
        box.setRange(-1_000_000_000.0, 1_000_000_000.0)
        box.setDecimals(6)
        box.setValue(float(value))
        return box

    def _operator(self, evidence: str, current: str):
        options = rule_operator_options(evidence)
        selected = current if current in options else options[0]
        return self._combo(
            [(item, OPERATOR_LABELS[item]) for item in options], selected
        )

    def _value(self, evidence: str, current):
        if is_categorical_evidence(evidence):
            values = rule_value_options(evidence)
            selected = str(current).upper() if current is not None else values[0]
            return self._combo([(item, _humanize(item)) for item in values], selected)
        return self._number(float(current))

    def _upper(self, evidence: str, current):
        if is_categorical_evidence(evidence):
            label = QLabel("—")
            label.setEnabled(False)
            return label
        return self._number(float(current))

    def _connect_control(self, widget) -> None:
        if isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(lambda *_args: self.changed.emit())
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(lambda *_args: self.changed.emit())

    def _install_rule_controls(self, row: int, rule: dict) -> None:
        evidence = rule["evidence"]
        operator = self._operator(evidence, rule["operator"])
        value = self._value(evidence, rule["value"])
        upper = self._upper(evidence, rule["value2"])
        self.setCellWidget(row, 1, operator)
        self.setCellWidget(row, 2, value)
        self.setCellWidget(row, 3, upper)
        self._connect_control(operator)
        self._connect_control(value)
        if isinstance(upper, QDoubleSpinBox):
            self._connect_control(upper)
        operator.currentIndexChanged.connect(
            lambda _index, current_row=row: self._refresh_upper(current_row)
        )
        self._refresh_upper(row)

    def _evidence_changed(self, row: int) -> None:
        evidence = self.cellWidget(row, 0)
        if not isinstance(evidence, QComboBox):
            return
        default = new_rule(kind=self.kind, evidence=evidence.currentData())
        self._install_rule_controls(row, default)
        self.changed.emit()

    def set_rules(self, rules) -> None:
        normalized = [
            normalize_rule(rule, expected_kind=self.kind) for rule in (rules or ())
        ]
        self.blockSignals(True)
        try:
            self.clearContents()
            self.setRowCount(len(normalized))
            self._ids = [rule["id"] for rule in normalized]
            for row, rule in enumerate(normalized):
                evidence = EvidenceComboBox(rule["evidence"])
                regime = self._combo(
                    [
                        ("ALL", "All Markets"),
                        *((item, item.title()) for item in REGIMES),
                    ],
                    rule["regime"],
                )
                side = self._combo(
                    [
                        ("ALL", "All Sides"),
                        *((item, item.title()) for item in SIDES),
                    ],
                    rule["side"],
                )
                self.setCellWidget(row, 0, evidence)
                self.setCellWidget(row, 4, regime)
                self.setCellWidget(row, 5, side)
                self._install_rule_controls(row, rule)
                self._connect_control(regime)
                self._connect_control(side)
                evidence.currentIndexChanged.connect(
                    lambda _index, current_row=row: self._evidence_changed(current_row)
                )
        finally:
            self.blockSignals(False)

    def _refresh_upper(self, row: int) -> None:
        evidence = self.cellWidget(row, 0)
        operator = self.cellWidget(row, 1)
        upper = self.cellWidget(row, 3)
        if not isinstance(evidence, QComboBox) or upper is None:
            return
        if is_categorical_evidence(evidence.currentData()):
            upper.setEnabled(False)
            return
        if isinstance(operator, QComboBox):
            upper.setEnabled(operator.currentData() in {"BETWEEN", "OUTSIDE"})

    def rules(self) -> tuple[dict, ...]:
        result = []
        for row in range(self.rowCount()):
            evidence = self.cellWidget(row, 0)
            operator = self.cellWidget(row, 1)
            value = self.cellWidget(row, 2)
            upper = self.cellWidget(row, 3)
            regime = self.cellWidget(row, 4)
            side = self.cellWidget(row, 5)
            evidence_name = evidence.currentData()
            categorical = is_categorical_evidence(evidence_name)
            result.append(
                normalize_rule(
                    {
                        "id": self._ids[row],
                        "kind": self.kind,
                        "evidence": evidence_name,
                        "operator": operator.currentData(),
                        "value": value.currentData() if categorical else value.value(),
                        "value2": None if categorical else upper.value(),
                        "regime": regime.currentData(),
                        "side": side.currentData(),
                    },
                    expected_kind=self.kind,
                )
            )
        return tuple(result)

    def add_rule(self) -> None:
        rules = list(self.rules())
        rules.append(new_rule(kind=self.kind))
        self.set_rules(rules)
        self.changed.emit()

    def remove_selected(self) -> None:
        rows = sorted(
            {index.row() for index in self.selectedIndexes()}, reverse=True
        )
        if not rows:
            return
        rules = list(self.rules())
        for row in rows:
            rules.pop(row)
        self.set_rules(rules)
        self.changed.emit()


class RuleStrategyBuilder(QWidget):
    """One strategy thesis expressed as permissions + scoped rules."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        summary_box = QGroupBox("Strategy Summary")
        summary_layout = QHBoxLayout(summary_box)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            "font-weight:600; background:#f7f9fb; padding:8px; border:1px solid #d9e2ec"
        )
        summary_layout.addWidget(self.summary, 1)
        layout.addWidget(summary_box)

        direction_box = QGroupBox("1. Direction & Market Eligibility")
        direction_layout = QVBoxLayout(direction_box)
        direction_form = QFormLayout()
        self.direction_mode = QComboBox()
        for mode in DIRECTION_MODES:
            self.direction_mode.addItem(DIRECTION_LABELS[mode], mode)
        direction_form.addRow("Direction strategy", self.direction_mode)
        direction_layout.addLayout(direction_form)

        permission = QGridLayout()
        permission.addWidget(QLabel("Market state"), 0, 0)
        permission.addWidget(QLabel("LONG"), 0, 1)
        permission.addWidget(QLabel("SHORT"), 0, 2)
        self.permission_checks: dict[str, QCheckBox] = {}
        for row, regime in enumerate(REGIMES, 1):
            permission.addWidget(QLabel(regime.title()), row, 0)
            for column, side in enumerate(SIDES, 1):
                key = f"{regime}_{side}"
                check = QCheckBox("Trade")
                check.setChecked(True)
                self.permission_checks[key] = check
                permission.addWidget(check, row, column)
                check.toggled.connect(lambda _checked: self._notify())
        direction_layout.addLayout(permission)
        note = QLabel(
            "These are permissions only. DI Direction uses raw +DI/-DI side selection. "
            "DMI Trend keeps that side selection and adds built-in ADX ≥ 20, non-falling ADX, "
            "and expanding DI pressure. The grid still decides where that side may trade; for the "
            "first clean trend test, enable Bull/Bear and leave Sideways off."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#52606d")
        direction_layout.addWidget(note)
        layout.addWidget(direction_box)

        required_box = QGroupBox("2. Entry Rules — all applicable rules must pass")
        required_layout = QVBoxLayout(required_box)
        self.required_rules = RuleTable("REQUIRED")
        required_layout.addWidget(self.required_rules)
        row = QHBoxLayout()
        add = QPushButton("+ Add Entry Rule")
        remove = QPushButton("Remove Selected")
        add.clicked.connect(self.required_rules.add_rule)
        remove.clicked.connect(self.required_rules.remove_selected)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        required_layout.addLayout(row)
        evidence_note = QLabel(
            "Evidence is grouped and searchable; common S/R choices are shown before advanced S/R details. OI, Funding, Basis and Taker Flow use causal prepared research when local coverage exists. A REQUIRED rule rejects a trade when its evidence is missing; missing VETO evidence does not create a rejection. Any S/R rule automatically enables causal S/R calculation; configure its calculation settings on Research Features."
        )
        evidence_note.setWordWrap(True)
        evidence_note.setStyleSheet("color:#52606d")
        required_layout.addWidget(evidence_note)
        layout.addWidget(required_box)

        veto_box = QGroupBox("3. Avoid / Veto Rules — matching conditions reject the trade")
        veto_layout = QVBoxLayout(veto_box)
        self.veto_rules = RuleTable("VETO")
        veto_layout.addWidget(self.veto_rules)
        row = QHBoxLayout()
        add = QPushButton("+ Add Veto Rule")
        remove = QPushButton("Remove Selected")
        add.clicked.connect(self.veto_rules.add_rule)
        remove.clicked.connect(self.veto_rules.remove_selected)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        veto_layout.addLayout(row)
        layout.addWidget(veto_box)

        research_box = QGroupBox("4. Research-only Evidence")
        research_layout = QHBoxLayout(research_box)
        self.enable_mr = QCheckBox("Attach Mean Reversion context")
        self.enable_mr.setChecked(True)
        research_layout.addWidget(self.enable_mr)
        self.research_status = QLabel(
            "S/R · OI · Funding · Positioning/Basis · Taker Flow are rule-ready. Detailed Trade Flow · Order Book remain Analyze Only until dedicated rule dependencies are added."
        )
        self.research_status.setWordWrap(True)
        self.research_status.setStyleSheet("color:#52606d")
        research_layout.addWidget(self.research_status, 1)
        layout.addWidget(research_box)

        self.show_advanced = QCheckBox("Show advanced direction actions and entry timing")
        layout.addWidget(self.show_advanced)
        self.advanced = QGroupBox("5. Advanced")
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_note = QLabel(
            "Direction flip rules are explicit conditional actions. Entry timing is separate from evidence filters."
        )
        advanced_note.setWordWrap(True)
        advanced_note.setStyleSheet("color:#52606d")
        advanced_layout.addWidget(advanced_note)
        self.flip_rules = RuleTable("FLIP")
        advanced_layout.addWidget(self.flip_rules)
        row = QHBoxLayout()
        add = QPushButton("+ Add Direction Flip Rule")
        remove = QPushButton("Remove Selected")
        add.clicked.connect(self.flip_rules.add_rule)
        remove.clicked.connect(self.flip_rules.remove_selected)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        advanced_layout.addLayout(row)

        timing_form = QFormLayout()
        self.entry_mode = QComboBox()
        self.entry_mode.addItem("Wait until current trade closes", "WAIT_UNTIL_CLOSED")
        self.entry_mode.addItem("Every N strategy candles", "EVERY_N_CANDLES")
        self.entry_interval = QSpinBox()
        self.entry_interval.setRange(1, 1_000_000)
        self.entry_interval.setValue(1)
        self.enable_daily_schedule = QCheckBox()
        self.daily_entry_time = QComboBox()
        for hour in range(24):
            self.daily_entry_time.addItem(f"{hour:02d}:00", f"{hour:02d}:00")
        self.daily_entry_timezone = QComboBox()
        self.daily_entry_timezone.setEditable(True)
        self.daily_entry_timezone.addItem("UTC", "UTC")
        self.daily_missed_policy = QComboBox()
        self.daily_missed_policy.addItem("Skip day", "SKIP_DAY")
        self.daily_missed_policy.addItem("Next available candle", "NEXT_AVAILABLE_CANDLE")
        self.momentum_lookback_hours = QSpinBox()
        self.momentum_lookback_hours.setRange(1, 87600)
        self.momentum_lookback_hours.setValue(24)
        self.momentum_lookback_hours.setSuffix(" h")
        timing_form.addRow("Entry cadence", self.entry_mode)
        timing_form.addRow("Entry interval", self.entry_interval)
        timing_form.addRow("Daily schedule", self.enable_daily_schedule)
        timing_form.addRow("Daily entry time", self.daily_entry_time)
        timing_form.addRow("Timezone", self.daily_entry_timezone)
        timing_form.addRow("Missed schedule", self.daily_missed_policy)
        timing_form.addRow("Momentum rule lookback", self.momentum_lookback_hours)
        advanced_layout.addLayout(timing_form)
        self.advanced.setVisible(False)
        self.show_advanced.toggled.connect(self.advanced.setVisible)
        layout.addWidget(self.advanced)
        layout.addStretch()

        for widget in (
            self.direction_mode,
            self.enable_mr,
            self.entry_mode,
            self.entry_interval,
            self.enable_daily_schedule,
            self.daily_entry_time,
            self.daily_entry_timezone,
            self.daily_missed_policy,
            self.momentum_lookback_hours,
        ):
            signal = (
                widget.toggled
                if isinstance(widget, QCheckBox)
                else widget.valueChanged
                if isinstance(widget, (QSpinBox, QDoubleSpinBox))
                else widget.currentIndexChanged
            )
            signal.connect(lambda *_args: self._notify())
        for table in (self.required_rules, self.veto_rules, self.flip_rules):
            table.changed.connect(self._notify)
        self._notify()

    def _notify(self):
        self.refresh_summary()
        self.changed.emit()

    def market_permissions(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in MARKET_PERMISSIONS
            if self.permission_checks[key].isChecked()
        )

    def refresh_summary(self):
        markets = []
        permissions = set(self.market_permissions())
        for regime in REGIMES:
            sides = [
                side.title()
                for side in SIDES
                if f"{regime}_{side}" in permissions
            ]
            markets.append(f"{regime.title()} {'/'.join(sides) if sides else 'Off'}")
        self.summary.setText(
            f"{DIRECTION_LABELS[self.direction_mode.currentData()]}  ·  "
            f"{' · '.join(markets)}  ·  "
            f"{len(self.required_rules.rules())} entry rule(s)  ·  "
            f"{len(self.veto_rules.rules())} veto rule(s)"
        )

    def set_from_strategy(self, strategy) -> None:
        profiles = strategy.profiles
        mode = infer_direction_mode(profiles)
        self.direction_mode.setCurrentIndex(
            max(0, self.direction_mode.findData(mode))
        )
        enabled = set(infer_market_permissions(profiles, mode))
        for key, check in self.permission_checks.items():
            check.setChecked(key in enabled)
        recovered = decompile_rules(profiles)
        self.required_rules.set_rules(recovered["REQUIRED"])
        self.veto_rules.set_rules(recovered["VETO"])
        self.flip_rules.set_rules(recovered["FLIP"])

        self.enable_mr.setChecked(bool(strategy.enable_mean_reversion_analysis))
        self.entry_mode.setCurrentIndex(
            max(0, self.entry_mode.findData(strategy.entry_mode))
        )
        self.entry_interval.setValue(int(strategy.entry_interval))
        self.enable_daily_schedule.setChecked(
            bool(strategy.enable_daily_entry_schedule)
        )
        time_index = self.daily_entry_time.findData(strategy.daily_entry_time)
        if time_index < 0:
            self.daily_entry_time.addItem(
                strategy.daily_entry_time, strategy.daily_entry_time
            )
            time_index = self.daily_entry_time.count() - 1
        self.daily_entry_time.setCurrentIndex(time_index)
        tz_index = self.daily_entry_timezone.findData(strategy.daily_entry_timezone)
        if tz_index < 0:
            self.daily_entry_timezone.addItem(
                strategy.daily_entry_timezone, strategy.daily_entry_timezone
            )
            tz_index = self.daily_entry_timezone.count() - 1
        self.daily_entry_timezone.setCurrentIndex(tz_index)
        self.daily_missed_policy.setCurrentIndex(
            max(
                0,
                self.daily_missed_policy.findData(
                    strategy.daily_entry_missed_policy
                ),
            )
        )
        if profiles:
            first = next(iter(profiles.values()))
            self.momentum_lookback_hours.setValue(
                int(first.momentum_lookback_hours)
            )
        self.refresh_summary()

    def strategy_values(self) -> dict:
        return {
            "direction_mode": self.direction_mode.currentData(),
            "market_permissions": self.market_permissions(),
            "required_rules": self.required_rules.rules(),
            "veto_rules": self.veto_rules.rules(),
            "flip_rules": self.flip_rules.rules(),
            "momentum_lookback_hours": self.momentum_lookback_hours.value(),
            "enable_di_direction_selection": True,
            # DI pressure stays available as causal evidence/reporting. The old
            # global allow-list is deliberately neutral; Entry/Veto rules own any
            # effect on trading decisions.
            "enable_di_pressure_analysis": True,
            "di_pressure_allow_expanding": True,
            "di_pressure_allow_contracting": True,
            "di_pressure_allow_mixed": True,
            "enable_mean_reversion_analysis": self.enable_mr.isChecked(),
            # S/R presets are retired from the rule-based runtime. S/R can still
            # be calculated for research, but entry effects belong only to rules.
            "sr_filter_mode": "ANALYSIS_ONLY",
            "sr_long_avoid_near_resistance": False,
            "sr_long_require_near_support": False,
            "sr_long_block_broken_support": False,
            "sr_long_min_room_to_resistance_atr": 0.0,
            "sr_short_avoid_near_support": False,
            "sr_short_require_near_resistance": False,
            "sr_short_block_broken_resistance": False,
            "sr_short_min_room_to_support_atr": 0.0,
            "entry_mode": self.entry_mode.currentData(),
            "entry_interval": self.entry_interval.value(),
            "enable_daily_entry_schedule": self.enable_daily_schedule.isChecked(),
            "daily_entry_time": self.daily_entry_time.currentData(),
            "daily_entry_timezone": (
                self.daily_entry_timezone.currentData()
                or self.daily_entry_timezone.currentText()
            ),
            "daily_entry_missed_policy": self.daily_missed_policy.currentData(),
        }

    def set_feature_status(self, features) -> None:
        items = [
            "S/R ON" if features.enable_support_resistance_analysis else "S/R Off",
            "OI",
            "Funding",
            "Positioning/Basis",
            "Taker Flow",
        ]
        items.append(
            "Trade Flow ON" if features.trade_flow_enabled else "Trade Flow Off"
        )
        items.append(
            "Order Book ON" if features.order_book_enabled else "Order Book Off"
        )
        self.research_status.setText(
            " · ".join(items)
            + " — S/R and lightweight futures context are rule-ready; detailed Trade Flow and Order Book remain research-only."
        )
