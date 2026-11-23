import { ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown } from "lucide-react";

export interface StudioSelectOption {
  value: string;
  label: string;
  description?: string;
  badge?: string;
  icon?: ReactNode;
  disabled?: boolean;
}

interface StudioSelectProps {
  value: string;
  options: StudioSelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  prefix?: string;
  leadingIcon?: ReactNode;
  title?: string;
  className?: string;
  compact?: boolean;
  disabled?: boolean;
  showTriggerBadge?: boolean;
  ariaLabel?: string;
}

export function StudioSelect({
  value,
  options,
  onChange,
  placeholder = "Chọn...",
  prefix,
  leadingIcon,
  title,
  className = "",
  compact = false,
  disabled = false,
  showTriggerBadge = true,
  ariaLabel,
}: StudioSelectProps) {
  const [open, setOpen] = useState(false);
  const [menuStyle, setMenuStyle] = useState<React.CSSProperties>({});
  const [openUp, setOpenUp] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const selected = useMemo(
    () => options.find((o) => o.value === value) ?? options[0] ?? null,
    [options, value]
  );

  const updatePosition = () => {
    const el = triggerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const margin = 8;
    const spaceBelow = window.innerHeight - rect.bottom - margin;
    const spaceAbove = rect.top - margin;
    const estHeight = Math.min(320, options.length * 64 + 16);
    const shouldOpenUp = spaceBelow < estHeight && spaceAbove > spaceBelow;
    setOpenUp(shouldOpenUp);

    const maxHeight = Math.max(120, Math.min(320, shouldOpenUp ? spaceAbove - 6 : spaceBelow - 6));
    const width = Math.max(rect.width, compact ? 180 : 252);
    const left = Math.max(margin, Math.min(rect.left, window.innerWidth - width - margin));

    if (shouldOpenUp) {
      setMenuStyle({
        position: "fixed",
        left,
        bottom: window.innerHeight - rect.top + 5,
        width,
        maxHeight,
      });
    } else {
      setMenuStyle({
        position: "fixed",
        left,
        top: rect.bottom + 5,
        width,
        maxHeight,
      });
    }
  };

  useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
  }, [open, options.length]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (
        (triggerRef.current && triggerRef.current.contains(target)) ||
        (menuRef.current && menuRef.current.contains(target))
      ) {
        return;
      }
      setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    const onScrollOrResize = (e: Event) => {
      if (menuRef.current && e.target instanceof Node && menuRef.current.contains(e.target)) {
        return;
      }
      updatePosition();
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onScrollOrResize);
    window.addEventListener("scroll", onScrollOrResize, true);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onScrollOrResize);
      window.removeEventListener("scroll", onScrollOrResize, true);
    };
  }, [open]);

  const activeIcon = selected?.icon ?? leadingIcon;

  return (
    <div className={`studio-select-wrap ${compact ? "compact" : ""} ${className}`}>
      <button
        ref={triggerRef}
        type="button"
        className={`studio-select-trigger ${open ? "open" : ""}`}
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        title={title || (selected ? `${selected.label}${selected.description ? ` — ${selected.description}` : ""}` : undefined)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel || title}
      >
        <span className="studio-select-main">
          {activeIcon && <span className="studio-select-icon">{activeIcon}</span>}
          {prefix && <span className="studio-select-prefix">{prefix}</span>}
          <span className="studio-select-label">{selected?.label || placeholder}</span>
        </span>
        <span className="studio-select-right">
          {showTriggerBadge && selected?.badge && (
            <span className="studio-select-badge">{selected.badge}</span>
          )}
          <ChevronDown size={13.5} className={`studio-select-chev ${open ? "open" : ""}`} />
        </span>
      </button>

      {open &&
        createPortal(
          <div
            ref={menuRef}
            className={`studio-select-menu ${openUp ? "open-up" : ""}`}
            style={menuStyle}
            role="listbox"
          >
            {options.map((opt) => {
              const isSelected = opt.value === selected?.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  disabled={opt.disabled}
                  className={`studio-select-option ${isSelected ? "selected" : ""}`}
                  onClick={() => {
                    if (opt.disabled) return;
                    onChange(opt.value);
                    setOpen(false);
                  }}
                >
                  <span className="studio-select-opt-left">
                    {opt.icon && <span className="studio-select-opt-icon">{opt.icon}</span>}
                    <span className="studio-select-opt-text">
                      <span className="studio-select-opt-top">
                        <span className="studio-select-opt-label">{opt.label}</span>
                        {opt.badge && <span className="studio-select-opt-badge">{opt.badge}</span>}
                      </span>
                      {opt.description && (
                        <span className="studio-select-opt-desc">{opt.description}</span>
                      )}
                    </span>
                  </span>
                  <span className="studio-select-opt-check">
                    {isSelected && <Check size={13} />}
                  </span>
                </button>
              );
            })}
          </div>,
          document.body
        )}
    </div>
  );
}
