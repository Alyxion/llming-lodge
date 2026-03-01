/**
 * Shared popup/window utilities for chat UI.
 *
 * Loaded Phase 2 (before ai-edit-shared.js).
 * Provides positioning, click-outside dismiss, and drag support.
 */
(function () {
  'use strict';

  /**
   * Position a fixed popup, clamping to viewport edges.
   * @param {HTMLElement} popup
   * @param {number} x - desired left
   * @param {number} y - desired top
   * @param {{margin?: number}} [opts]
   */
  function cv2PositionPopup(popup, x, y, opts) {
    var margin = (opts && opts.margin) || 8;
    popup.style.position = 'fixed';
    popup.style.left = x + 'px';
    popup.style.top = y + 'px';
    // Reposition after render to clamp within viewport
    requestAnimationFrame(function () {
      var rect = popup.getBoundingClientRect();
      if (rect.right > window.innerWidth - margin) {
        popup.style.left = Math.max(margin, window.innerWidth - rect.width - margin) + 'px';
      }
      if (rect.bottom > window.innerHeight - margin) {
        popup.style.top = Math.max(margin, window.innerHeight - rect.height - margin) + 'px';
      }
    });
  }

  /**
   * Dismiss a popup on click-outside or Escape.
   * @param {HTMLElement} popup
   * @param {function} onClose - called when dismissed
   * @returns {function} cleanup - call to remove listeners
   */
  function cv2DismissOnOutside(popup, onClose, excludeEl) {
    function onKey(e) {
      if (e.key === 'Escape') { cleanup(); onClose(); }
    }
    function onClick(e) {
      if (!popup.contains(e.target) && !(excludeEl && excludeEl.contains(e.target))) {
        cleanup(); onClose();
      }
    }
    function cleanup() {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onClick);
    }
    // Delay to avoid the triggering click itself
    setTimeout(function () {
      document.addEventListener('keydown', onKey);
      document.addEventListener('mousedown', onClick);
    }, 50);
    return cleanup;
  }

  /**
   * Make an element draggable by its handle.
   * @param {HTMLElement} el - the element to move
   * @param {HTMLElement} handleEl - the drag handle
   * @returns {function} cleanup - call to remove listeners
   */
  function cv2MakeDraggable(el, handleEl) {
    var startX, startY, origLeft, origTop;
    var dragging = false;

    function onMouseDown(e) {
      if (e.button !== 0) return;
      e.preventDefault();
      dragging = true;
      var rect = el.getBoundingClientRect();
      startX = e.clientX;
      startY = e.clientY;
      origLeft = rect.left;
      origTop = rect.top;
      handleEl.style.cursor = 'grabbing';
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    }

    function onMouseMove(e) {
      if (!dragging) return;
      var dx = e.clientX - startX;
      var dy = e.clientY - startY;
      var newLeft = origLeft + dx;
      var newTop = origTop + dy;
      // Constrain to viewport
      var rect = el.getBoundingClientRect();
      newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - rect.width));
      newTop = Math.max(0, Math.min(newTop, window.innerHeight - rect.height));
      el.style.left = newLeft + 'px';
      el.style.top = newTop + 'px';
    }

    function onMouseUp() {
      dragging = false;
      handleEl.style.cursor = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    }

    handleEl.addEventListener('mousedown', onMouseDown);

    return function cleanup() {
      handleEl.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }

  /**
   * Check if chat app is in dark mode.
   * @returns {boolean}
   */
  function cv2IsDark() {
    var el = document.getElementById('chat-app');
    return !!(el && el.classList.contains('cv2-dark'));
  }

  /** Escape text for safe innerHTML insertion. */
  function _escText(s) {
    if (!s) return '';
    var d = document.createElement('span');
    d.textContent = s;
    return d.innerHTML;
  }

  /**
   * Build a single popup item (button).
   * @param {Object} item - { icon?, flag?, label, action?, toggle?, active?, noClose? }
   * @param {function} onAction - called with (action, item)
   * @param {function} dismissFn - closes the menu
   * @returns {HTMLElement}
   */
  function _buildPopupItem(item, onAction, dismissFn) {
    var row = document.createElement('button');
    row.type = 'button';
    row.className = 'cv2-popup-item';
    if (item.action) row.dataset.action = item.action;
    var iconHtml = '';
    if (item.flag) {
      iconHtml = '<span class="cv2-popup-flag">' + item.flag + '</span>';
    } else if (item.icon) {
      var style = 'font-size:16px';
      if (item.toggle && item.active) style += ';color:#4ade80';
      iconHtml = '<span class="material-icons" style="' + style + '">' + item.icon + '</span>';
    }
    row.innerHTML = iconHtml + ' ' + _escText(item.label);
    row.addEventListener('mousedown', function (e) { e.preventDefault(); });
    row.addEventListener('click', function () {
      if (onAction) onAction(item.action || null, item);
      if (!item.noClose) dismissFn();
    });
    return row;
  }

  /**
   * Create and show a styled popup menu.
   * Unified builder — use this for all popup menus to ensure consistent
   * styling, dark mode, and boundary handling.
   *
   * @param {Array} items - menu items:
   *   { icon?, flag?, label, action?, submenu?: Array, toggle?, active?, noClose? }
   *   or '---' for a separator
   * @param {Object} opts
   * @param {number} opts.x - left position
   * @param {number} opts.y - top position
   * @param {number} [opts.minWidth=180] - min-width in px
   * @param {function} [opts.onAction] - called with (action, item) on item click
   * @param {function} [opts.onDismiss] - called when menu is dismissed
   * @returns {{ el: HTMLElement, dismiss: function }}
   */
  function cv2PopupMenu(items, opts) {
    opts = opts || {};
    var isDark = cv2IsDark();

    var menu = document.createElement('div');
    menu.className = 'cv2-rich-popup cv2-popup-menu' + (isDark ? ' cv2-dark' : '');
    if (opts.minWidth) menu.style.minWidth = opts.minWidth + 'px';

    function dismiss() {
      _cleanup();
      if (menu.parentNode) menu.remove();
      if (opts.onDismiss) opts.onDismiss();
    }

    for (var i = 0; i < items.length; i++) {
      var item = items[i];

      // Separator
      if (item === '---') {
        var sep = document.createElement('div');
        sep.className = 'cv2-popup-sep';
        menu.appendChild(sep);
        continue;
      }

      // Submenu
      if (item.submenu) {
        var subWrap = document.createElement('div');
        subWrap.className = 'cv2-popup-sub';
        subWrap.tabIndex = 0;
        var trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'cv2-popup-item cv2-popup-sub-trigger';
        var tIcon = item.icon
          ? '<span class="material-icons" style="font-size:16px">' + item.icon + '</span> '
          : '';
        trigger.innerHTML = tIcon + _escText(item.label) +
          ' <span class="material-icons cv2-popup-chevron">chevron_right</span>';
        subWrap.appendChild(trigger);

        var subPanel = document.createElement('div');
        subPanel.className = 'cv2-popup-submenu cv2-rich-popup' + (isDark ? ' cv2-dark' : '');
        for (var j = 0; j < item.submenu.length; j++) {
          subPanel.appendChild(_buildPopupItem(item.submenu[j], opts.onAction, dismiss));
        }
        subWrap.appendChild(subPanel);
        menu.appendChild(subWrap);

        // Boundary detection: flip submenu if it would overflow
        (function (sw, sp) {
          sw.addEventListener('mouseenter', function () {
            sp.style.left = '100%'; sp.style.right = '';
            sp.style.top = '-4px'; sp.style.bottom = '';
            requestAnimationFrame(function () {
              var r = sp.getBoundingClientRect();
              if (r.right > window.innerWidth - 8) {
                sp.style.left = 'auto'; sp.style.right = '100%';
              }
              if (r.bottom > window.innerHeight - 8) {
                sp.style.top = 'auto'; sp.style.bottom = '0';
              }
            });
          });
        })(subWrap, subPanel);
        continue;
      }

      // Regular item
      menu.appendChild(_buildPopupItem(item, opts.onAction, dismiss));
    }

    document.body.appendChild(menu);
    cv2PositionPopup(menu, opts.x || 0, opts.y || 0);

    var _cleanup = cv2DismissOnOutside(menu, function () { dismiss(); }, opts.anchor || null);

    return { el: menu, dismiss: dismiss };
  }

  /**
   * Show a styled prompt dialog (replaces window.prompt).
   * @param {string} message - prompt label
   * @param {Object} [opts]
   * @param {string} [opts.placeholder] - input placeholder
   * @param {string} [opts.confirmLabel='OK']
   * @param {string} [opts.cancelLabel='Cancel']
   * @returns {Promise<string|null>} - input value or null if cancelled
   */
  function cv2PromptDialog(message, opts) {
    opts = opts || {};
    var isDark = cv2IsDark();
    return new Promise(function (resolve) {
      var backdrop = document.createElement('div');
      backdrop.className = 'cv2-prompt-backdrop';

      var dialog = document.createElement('div');
      dialog.className = 'cv2-prompt-dialog' + (isDark ? ' cv2-dark' : '');
      dialog.innerHTML =
        '<div class="cv2-prompt-label">' + _escText(message) + '</div>' +
        '<input type="text" class="cv2-prompt-input" placeholder="' +
          (opts.placeholder ? _escText(opts.placeholder) : '') + '" />' +
        '<div class="cv2-prompt-actions">' +
          '<button type="button" class="cv2-prompt-btn cv2-prompt-cancel">' +
            _escText(opts.cancelLabel || 'Cancel') + '</button>' +
          '<button type="button" class="cv2-prompt-btn cv2-prompt-ok">' +
            _escText(opts.confirmLabel || 'OK') + '</button>' +
        '</div>';

      backdrop.appendChild(dialog);
      document.body.appendChild(backdrop);

      var input = dialog.querySelector('.cv2-prompt-input');
      input.focus();

      var resolved = false;
      function close(value) {
        if (resolved) return;
        resolved = true;
        backdrop.remove();
        resolve(value);
      }

      dialog.querySelector('.cv2-prompt-ok').addEventListener('click', function () {
        close(input.value.trim() || null);
      });
      dialog.querySelector('.cv2-prompt-cancel').addEventListener('click', function () {
        close(null);
      });
      backdrop.addEventListener('click', function (e) {
        if (e.target === backdrop) close(null);
      });
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); close(input.value.trim() || null); }
        if (e.key === 'Escape') { e.preventDefault(); close(null); }
      });
    });
  }

  // Expose globally
  window.cv2PositionPopup = cv2PositionPopup;
  window.cv2DismissOnOutside = cv2DismissOnOutside;
  window.cv2MakeDraggable = cv2MakeDraggable;
  window.cv2IsDark = cv2IsDark;
  window.cv2PopupMenu = cv2PopupMenu;
  window.cv2PromptDialog = cv2PromptDialog;

})();
